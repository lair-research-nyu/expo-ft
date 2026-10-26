"""CLI flags and construction for the EXPO-FT runtime inside server_openpi.

Kept here, not in the server, so policy-inference-spec only needs a handful of lines:
add_cli_args(parser) in _parse_args and build_runtime(args, policy) in _run_cli.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from expo_ft.ultra.config import ExpoFtConfig
from expo_ft.ultra.rewards import FixedLabel, HumanLabel, Outcome

LOGGER = logging.getLogger(__name__)
DEFAULT_RUN_ROOT = Path("~/exp/expo-ft")


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("EXPO-FT (RL fine-tuning: critic + edit policy over a frozen VLA)")
    g.add_argument("--expo-ft", action="store_true", help="Serve the EXPO-FT on-the-fly policy and train online. "
                   "Requires --rtc-train.")
    g.add_argument("--expo-ft-config", type=Path, default=None,
                   help="JSON file of ExpoFtConfig fields to override.")
    g.add_argument("--expo-ft-set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="ExpoFtConfig overrides, e.g. exec_steps=20 edit_scale_arm=0.05 (JSON values).")
    g.add_argument("--expo-ft-run-dir", type=Path, default=None,
                   help=f"Metrics, checkpoints and episode files (default: {DEFAULT_RUN_ROOT}/<timestamp>).")
    g.add_argument("--expo-ft-label", choices=("human", "reason", "success", "failure", "abort"), default="human",
                   help="Episode labels: the operator's keypress (default); 'reason' takes the client's "
                        "done_reason when it is success/failure/abort (dry runs); or a fixed label.")
    g.add_argument("--expo-ft-base-only", action="store_true",
                   help="Execute the first base sample (no Q selection, no edits) while still recording and "
                        "training: a baseline / seam comparison mode.")
    g.add_argument("--expo-ft-demos", type=Path, default=None, help="LeRobot v3 dataset root to seed the buffer.")
    g.add_argument("--expo-ft-num-demos", type=int, default=20, help="Demonstrations to seed (paper: 10-40).")
    g.add_argument("--expo-ft-demo-task", default=None, help="Only demos of this task (default: --task).")
    g.add_argument("--expo-ft-demo-max-len", type=int, default=1500, help="Skip demos longer than this (frames).")
    g.add_argument("--expo-ft-resume", type=Path, default=None, help="A saved state_*.msgpack to start from.")
    g.add_argument("--expo-ft-resume-episodes", type=Path, default=None,
                   help="A previous run's episodes/ directory to reload into the buffer.")


def config_from_args(args: argparse.Namespace) -> ExpoFtConfig:
    over: dict[str, Any] = {}
    if args.expo_ft_config is not None:
        over.update(json.loads(Path(args.expo_ft_config).read_text()))
    for kv in args.expo_ft_set:
        key, _, value = kv.partition("=")
        try:
            over[key] = json.loads(value)
        except json.JSONDecodeError:
            over[key] = value
    for k in ("camera_keys", "hidden_dims", "encoder_stage_sizes"):
        if k in over:
            over[k] = tuple(over[k])
    over.setdefault("horizon", args.model_action_horizon)
    return ExpoFtConfig(**over)


def build_runtime(args: argparse.Namespace, policy: Any, *, predict_lock=None, norm_stats_dir: Path):
    """`policy` is server_openpi's OpenpiChunkPolicy (its openpi Policy is policy._policy)."""
    from expo_ft.ultra.runtime import ExpoRuntime, load_episode

    assert args.rtc_train, "--expo-ft needs --rtc-train: candidates are sampled with the committed prefix pinned"
    cfg = config_from_args(args)
    assert cfg.max_prefix <= policy.rtc_train_max_delay, (cfg.max_prefix, policy.rtc_train_max_delay)
    run_dir = args.expo_ft_run_dir or DEFAULT_RUN_ROOT.expanduser() / time.strftime("%Y_%m_%d_%H_%M_%S")
    if args.expo_ft_label == "human":
        labeler = HumanLabel()
    elif args.expo_ft_label == "reason":
        labeler = FixedLabel(Outcome.ABORT, by_reason={o.value: o for o in Outcome})
    else:
        labeler = FixedLabel(Outcome(args.expo_ft_label))
    runtime = ExpoRuntime(policy._policy, cfg, norm_stats_dir=norm_stats_dir, run_dir=run_dir,  # noqa: SLF001
                          num_steps=args.num_steps, labeler=labeler, predict_lock=predict_lock)
    runtime.base_only = args.expo_ft_base_only
    LOGGER.warning("EXPO-FT ON: run dir %s; config %s", run_dir, dataclasses.asdict(cfg))
    if args.expo_ft_resume is not None:
        runtime.load(args.expo_ft_resume)
    seeds = []
    if args.expo_ft_demos is not None:
        seeds += load_demos(args, cfg, runtime)
    if args.expo_ft_resume_episodes is not None:
        seeds += [load_episode(p) for p in sorted(Path(args.expo_ft_resume_episodes).glob("episode_*.npz"))]
    if seeds:
        seeds.sort(key=lambda e: not e.is_demo)  # demos first: they live in the protected region
        runtime.add_episodes(seeds)
    return runtime


def load_demos(args: argparse.Namespace, cfg: ExpoFtConfig, runtime) -> list:
    from expo_ft.ultra.demos import LeRobotV3, demo_episode

    reader = LeRobotV3(args.expo_ft_demos)
    task = args.expo_ft_demo_task or args.task
    ids = reader.episode_ids(task=task, exit_types=["success"])
    lengths = reader.episodes.loc[ids, "length"]
    ids = [i for i in ids if lengths.loc[i] <= args.expo_ft_demo_max_len]
    rng = np.random.default_rng(cfg.seed)
    chosen = sorted(rng.choice(ids, size=min(args.expo_ft_num_demos, len(ids)), replace=False).tolist())
    LOGGER.info("seeding %d demos of task %r from %s: %s", len(chosen), task, reader.root, chosen)
    return [demo_episode(reader, ep, cfg, observe=runtime.observation,
                         normalize_actions=runtime.action_norm.normalize, rng=rng, prompt=task)
            for ep in chosen]


def warmup(runtime) -> None:
    """Compile select and update before the robot connects."""
    import jax

    cfg = runtime.cfg
    h, w = 360, 640
    from expo_ft.ultra.runtime import CAMERA_TO_SLOT

    images = {slot: np.zeros((h, w, 3), np.uint8) for slot in CAMERA_TO_SLOT.values()}
    state = runtime.state_norm.unnormalize(np.zeros(cfg.action_dim, np.float32))
    t0 = time.perf_counter()
    runtime.predict(images, state, "warmup", prefix=None)
    runtime.predict(images, state, "warmup", prefix=np.repeat(state[None], 5, 0))
    runtime._pending = None
    runtime._decisions.clear()
    t1 = time.perf_counter()
    n = cfg.batch_size * cfg.utd_ratio
    v, s, a, c = len(cfg.camera_keys), cfg.image_size, cfg.action_dim, cfg.exec_steps
    batch = {
        "images": np.zeros((n, v, s, s, 3), np.uint8), "state": np.zeros((n, a), np.float32),
        "actions": np.zeros((n, c * a), np.float32), "base": np.zeros((n, c, a), np.float32),
        "prefix_len": np.zeros(n, np.int32), "rewards": np.zeros(n, np.float32),
        "discounts": np.zeros(n, np.float32), "next_images": np.zeros((n, v, s, s, 3), np.uint8),
        "next_state": np.zeros((n, a), np.float32), "next_base": np.zeros((n, cfg.n_base, c, a), np.float32),
        "next_prefix_len": np.zeros(n, np.int32),
    }
    jax.block_until_ready(runtime.learner.update(runtime.state, batch))  # result discarded
    LOGGER.info("EXPO-FT warmup: select %.1fs, update compile %.1fs", t1 - t0, time.perf_counter() - t1)
