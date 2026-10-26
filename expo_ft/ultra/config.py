"""EXPO-FT for the Ultra GEN2 (pi0.5 trained with train-time RTC, 25-D actions at 50 Hz).

Defaults are the paper's Table 3 (Appendix D.2) unless a comment says otherwise. Every
quantity that touches actions is in the policy's *normalized* action space (quantile
normalization from the checkpoint's norm stats): that is where the base chunks come out
of the flow, where edits are added, and what the critic sees.

See EXPO-IMPLEMENTATION.md at the repo root for the full design.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import numpy as np

# 25-D GEN2 action/state layout (server/wire.py _wire_state_25):
#   left arm 0:7, left gripper 7, right arm 8:15, right gripper 15, chest 16:22, neck 22:25.
ACTION_DIM = 25
LEFT_ARM = tuple(range(0, 7))
LEFT_GRIPPER = (7,)
RIGHT_ARM = tuple(range(8, 15))
RIGHT_GRIPPER = (15,)
CHEST = tuple(range(16, 22))
NECK = tuple(range(22, 25))
CAMERA_KEYS = ("main", "left_wrist", "right_wrist")


@dataclasses.dataclass(frozen=True)
class ExpoFtConfig:
    # --- chunking -----------------------------------------------------------------
    action_dim: int = ACTION_DIM
    horizon: int = 50  # H: rows the VLA predicts per chunk
    exec_steps: int = 20  # C: rows executed per decision (client "steps per inference")
    max_prefix: int = 15  # RTC delays the base policy was trained on: d in [0, 15)

    # --- OTF selection (paper: N base samples + N edits, argmax Q) -----------------
    n_base: int = 8
    n_edit: int = 8

    # --- edit policy --------------------------------------------------------------
    # One offset knot every `edit_knot_stride` rows of the executed window, linearly
    # interpolated; 1 = an independent offset on every row.
    edit_knot_stride: int = 5
    # Per-dimension edit scale in normalized units; 0 masks a dimension out entirely
    # (it is not even part of the edit distribution). Paper Table 4: 0.05-0.2.
    edit_scale_arm: float = 0.1
    edit_scale_gripper: float = 0.2
    edit_scale_chest: float = 0.0
    edit_scale_neck: float = 0.0
    # Rows after the prefix fade the edit in over this many rows (0 = step change at d).
    edit_ramp_rows: int = 5

    # --- networks (Appendix D.1) --------------------------------------------------
    image_size: int = 224
    camera_keys: tuple[str, ...] = CAMERA_KEYS
    encoder_stage_sizes: tuple[int, ...] = (3, 4, 6, 3)
    encoder_num_filters: int = 64
    image_latent_dim: int = 512  # per camera
    state_latent_dim: int = 64
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    num_qs: int = 10  # REDQ ensemble size
    num_min_qs: int = 2  # REDQ subset size for targets and selection

    # --- optimisation (Table 3) ---------------------------------------------------
    lr: float = 3e-4
    discount: float = 0.999  # per 50 Hz step; per decision it is discount ** C
    init_temperature: float = 1.0
    tau_critic: float = 5e-3
    tau_actor: float = 1e-3  # only used once VLA BC updates exist (phase 2)
    batch_size: int = 64
    utd_ratio: int = 20
    updates_per_episode: int = 4  # Table 4: 1-6
    target_entropy: Optional[float] = None  # None: -edit_dim / 2 (the authors' default)
    augment_pad: int = 8  # random-shift augmentation on critic images, pixels

    # --- data ---------------------------------------------------------------------
    buffer_capacity: int = 60_000  # decision-point transitions (~27 GB of images)
    demo_ratio: float = 0.0  # 0: demos sampled uniformly with online data
    # Reuse the N base samples drawn at s_{t+C} as the TD-target base candidates. Only
    # valid while the VLA is frozen; must be False once VLA updates are enabled.
    cache_base_candidates: bool = True
    # How a human "abort" label enters the buffer: "truncate" (keep, bootstrap the last
    # transition) or "discard" (drop the episode).
    abort_mode: str = "truncate"
    success_reward: float = 1.0
    failure_reward: float = 0.0
    # During a human takeover, keep an observation anchor at most every this many rows.
    hil_anchor_stride: Optional[int] = None  # None: exec_steps

    # --- phase 2 ------------------------------------------------------------------
    train_vla: bool = False

    seed: int = 0

    def __post_init__(self):
        assert 0 < self.exec_steps <= self.horizon
        assert self.max_prefix < self.exec_steps + 1, "prefix must fit inside the executed window"
        assert 1 <= self.edit_knot_stride <= self.exec_steps
        assert self.abort_mode in ("truncate", "discard"), self.abort_mode
        assert self.num_min_qs <= self.num_qs
        if self.train_vla:
            assert not self.cache_base_candidates, "cached base candidates go stale once the VLA trains"
        assert len(self.edit_dims) > 0, "every action dimension is masked out of the edit"

    # ---- derived quantities ----
    @property
    def edit_scale_vector(self) -> np.ndarray:
        """(action_dim,) per-dimension edit scale; 0 where masked."""
        s = np.zeros(self.action_dim, np.float32)
        s[list(LEFT_ARM + RIGHT_ARM)] = self.edit_scale_arm
        s[list(LEFT_GRIPPER + RIGHT_GRIPPER)] = self.edit_scale_gripper
        s[list(CHEST)] = self.edit_scale_chest
        s[list(NECK)] = self.edit_scale_neck
        return s

    @property
    def edit_dims(self) -> tuple[int, ...]:
        """Action dimensions the edit policy controls (non-zero scale)."""
        return tuple(int(i) for i in np.flatnonzero(self.edit_scale_vector > 0))

    @property
    def num_knots(self) -> int:
        return int(math.ceil((self.exec_steps - 1) / self.edit_knot_stride)) + 1

    @property
    def edit_param_dim(self) -> int:
        """Dimension of the tanh-Gaussian: knots x edited action dims."""
        return self.num_knots * len(self.edit_dims)

    @property
    def chunk_flat_dim(self) -> int:
        """Critic action input: the C executed rows, flattened."""
        return self.exec_steps * self.action_dim

    @property
    def decision_discount(self) -> float:
        return self.discount**self.exec_steps

    @property
    def n_candidates(self) -> int:
        return self.n_base + self.n_edit

    @property
    def resolved_target_entropy(self) -> float:
        return -self.edit_param_dim / 2 if self.target_entropy is None else self.target_entropy

    @property
    def resolved_hil_anchor_stride(self) -> int:
        return self.exec_steps if self.hil_anchor_stride is None else self.hil_anchor_stride
