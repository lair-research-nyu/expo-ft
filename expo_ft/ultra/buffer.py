"""Decision-point replay buffer.

Storage is per *observation slot*: one slot per decision (or human-takeover anchor, or
terminal observation). A transition lives in the slot of its source observation and points
at the slot of its bootstrap observation, so next-observation images are never stored
twice. Each slot also caches the N base chunks the frozen VLA produced at that
observation with that observation's RTC prefix pinned -- they are the TD-target base
candidates of the transition that ends there, and the base samples the edit policy is
trained to edit at that observation.

Episodes occupy contiguous slot ranges. Demos are stored first and never evicted; online
episodes fill a ring after them and evict whole episodes, oldest first.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from expo_ft.ultra.config import ExpoFtConfig


@dataclasses.dataclass
class Slot:
    """One stored observation (all actions/state normalized unless named *_robot)."""

    images: np.ndarray  # (V, S, S, 3) uint8, resize-with-pad of each camera
    state: np.ndarray  # (A,) normalized
    state_robot: np.ndarray  # (A,) robot space (the VLA's own input, for candidate fills)
    prefix_len: int = 0  # RTC prefix d pinned at this observation
    prefix_robot: Optional[np.ndarray] = None  # (d, A) robot-space committed rows
    candidates: Optional[np.ndarray] = None  # (N, C, A) base chunks, rows [0, C)


@dataclasses.dataclass
class Transition:
    src: int  # slot index within the episode
    next: int  # bootstrap slot index within the episode, -1 if none (then discount must be 0)
    actions: np.ndarray  # (C, A) normalized executed rows (policy and/or human)
    reward: float
    discount: float  # gamma ** n_executed * mask
    human_rows: int = 0  # rows of `actions` that came from a human (logging only)


@dataclasses.dataclass
class Episode:
    slots: list[Slot]
    transitions: list[Transition]
    prompt: str = ""
    is_demo: bool = False
    outcome: str = ""
    info: dict = dataclasses.field(default_factory=dict)

    def validate(self, cfg: ExpoFtConfig) -> None:
        srcs = [t.src for t in self.transitions]
        assert len(set(srcs)) == len(srcs), "at most one transition per source slot"
        for t in self.transitions:
            assert 0 <= t.src < len(self.slots)
            assert -1 <= t.next < len(self.slots) and t.next != t.src
            assert t.actions.shape == (cfg.exec_steps, cfg.action_dim), t.actions.shape
            assert t.next >= 0 or t.discount == 0.0, "a transition without a next observation must be terminal"
        for s in self.slots:
            assert 0 <= s.prefix_len < cfg.exec_steps
            assert s.prefix_len == 0 or len(s.prefix_robot) >= s.prefix_len, "prefix rows missing"
            if s.candidates is not None:
                assert s.candidates.shape == (cfg.n_base, cfg.exec_steps, cfg.action_dim), s.candidates.shape


class ReplayBuffer:
    def __init__(self, cfg: ExpoFtConfig, capacity: Optional[int] = None, seed: int = 0):
        self.cfg = cfg
        self.capacity = int(capacity or cfg.buffer_capacity)
        self.rng = np.random.default_rng(seed)
        n, v, s, a, c = self.capacity, len(cfg.camera_keys), cfg.image_size, cfg.action_dim, cfg.exec_steps
        # np.zeros of this size is lazily backed by the OS: untouched pages cost nothing.
        self.images = np.zeros((n, v, s, s, 3), np.uint8)
        self.state = np.zeros((n, a), np.float32)
        self.state_robot = np.zeros((n, a), np.float32)
        self.prefix_len = np.zeros(n, np.int32)
        self.prefix_robot = np.zeros((n, cfg.max_prefix, a), np.float32)
        self.candidates = np.zeros((n, cfg.n_base, c, a), np.float32)
        self.has_candidates = np.zeros(n, bool)
        self.has_transition = np.zeros(n, bool)
        self.next_slot = np.full(n, -1, np.int64)
        self.actions = np.zeros((n, c, a), np.float32)
        self.rewards = np.zeros(n, np.float32)
        self.discounts = np.zeros(n, np.float32)
        self.is_demo = np.zeros(n, bool)
        self.episode_of = np.full(n, -1, np.int64)
        self.episodes: dict[int, dict] = {}  # id -> {start, length, prompt, is_demo, outcome}
        self._next_id = 0
        self._demo_end = 0  # slots [0, demo_end) are demos
        self._cursor = 0  # next online write position (>= demo_end)

    # ------------------------------------------------------------------ writing
    def add_episode(self, ep: Episode) -> int:
        ep.validate(self.cfg)
        length = len(ep.slots)
        if ep.is_demo:
            assert not any(not e["is_demo"] for e in self.episodes.values()), "add demos before online data"
            start = self._demo_end
            assert start + length <= self.capacity, "demos alone exceed the buffer capacity"
            self._demo_end = self._cursor = start + length
        else:
            assert length <= self.capacity - self._demo_end, "episode longer than the online region"
            start = self._cursor
            if start + length > self.capacity:
                start = self._demo_end
            self._evict_range(start, start + length)
            self._cursor = start + length
        eid = self._next_id
        self._next_id += 1
        idx = np.arange(start, start + length)
        self.episode_of[idx] = eid
        self.is_demo[idx] = ep.is_demo
        self.has_transition[idx] = False
        self.next_slot[idx] = -1
        for i, s in zip(idx, ep.slots):
            self.images[i] = s.images
            self.state[i] = s.state
            self.state_robot[i] = s.state_robot
            self.prefix_len[i] = s.prefix_len
            self.prefix_robot[i] = 0.0
            if s.prefix_len:
                self.prefix_robot[i, : s.prefix_len] = s.prefix_robot[: s.prefix_len]
            self.has_candidates[i] = s.candidates is not None
            if s.candidates is not None:
                self.candidates[i] = s.candidates
        for t in ep.transitions:
            i = start + t.src
            self.has_transition[i] = True
            self.next_slot[i] = -1 if t.next < 0 else start + t.next
            self.actions[i] = t.actions
            self.rewards[i] = t.reward
            self.discounts[i] = t.discount
        self.episodes[eid] = {"start": start, "length": length, "prompt": ep.prompt,
                              "is_demo": ep.is_demo, "outcome": ep.outcome}
        return eid

    def _evict_range(self, lo: int, hi: int) -> None:
        for eid in np.unique(self.episode_of[lo:hi]):
            if eid < 0:
                continue
            e = self.episodes.pop(int(eid))
            assert not e["is_demo"]
            sl = slice(e["start"], e["start"] + e["length"])
            self.episode_of[sl] = -1
            self.has_transition[sl] = False
            self.has_candidates[sl] = False

    def set_candidates(self, slots: np.ndarray, candidates: np.ndarray) -> None:
        self.candidates[slots] = candidates
        self.has_candidates[slots] = True

    # ------------------------------------------------------------------ reading
    def __len__(self) -> int:
        return int(self.has_transition.sum())

    def missing_candidates(self) -> np.ndarray:
        """Live slots whose base candidates still have to be sampled: sources of a transition
        (edit-policy base) and bootstrap targets of a non-terminal one (TD candidates)."""
        live = self.episode_of >= 0
        need = live & self.has_transition
        boot = self.next_slot[self.has_transition & (self.discounts > 0)]
        need[boot] = True
        return np.flatnonzero(need & ~self.has_candidates)

    def sample(self, n: int) -> dict[str, np.ndarray]:
        """n transitions with everything ExpoLearner.update reads."""
        valid = np.flatnonzero(self.has_transition)
        assert len(valid), "empty buffer"
        ratio = self.cfg.demo_ratio
        demo = valid[self.is_demo[valid]]
        if ratio > 0 and len(demo) and len(demo) < len(valid):
            online = valid[~self.is_demo[valid]]
            k = self.rng.binomial(n, ratio)
            idx = np.concatenate([self.rng.choice(demo, k), self.rng.choice(online, n - k)])
        else:
            idx = self.rng.choice(valid, n)
        nxt = self.next_slot[idx]
        terminal = nxt < 0
        nxt_safe = np.where(terminal, idx, nxt)  # any valid slot; discount is 0 there
        assert np.all(self.has_candidates[nxt_safe] | (self.discounts[idx] == 0)), \
            "bootstrap slot without candidates: fill them before sampling"
        # Edit-policy base: one of the cached base samples at s, or the executed rows if none.
        j = self.rng.integers(0, self.cfg.n_base, n)
        base = np.where(self.has_candidates[idx][:, None, None], self.candidates[idx, j], self.actions[idx])
        return {
            "images": self.images[idx],
            "state": self.state[idx],
            "actions": self.actions[idx].reshape(n, -1),
            "base": base,
            "prefix_len": self.prefix_len[idx],
            "rewards": self.rewards[idx],
            "discounts": self.discounts[idx],
            "next_images": self.images[nxt_safe],
            "next_state": self.state[nxt_safe],
            "next_base": self.candidates[nxt_safe],
            "next_prefix_len": self.prefix_len[nxt_safe],
        }

    def stats(self) -> dict:
        eps = list(self.episodes.values())
        online = [e for e in eps if not e["is_demo"]]
        return {
            "transitions": len(self),
            "episodes_online": len(online),
            "episodes_demo": len(eps) - len(online),
            "online_success": sum(e["outcome"] == "success" for e in online),
        }
