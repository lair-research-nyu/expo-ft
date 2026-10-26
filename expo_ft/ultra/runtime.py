"""The EXPO-FT actor + learner living inside server_openpi (one process, one GPU).

The server keeps doing everything it already does -- wire validation, RTC prefix
resolution (_resolve_prefix), the action limiter, truncation, chunk_id echo -- and calls
in at four points:

    predict(...)        replaces policy.predict: N RTC candidates + N edits, argmax Q
    record_sent(chunk)  the chunk actually sent (after the limiter), for the buffer
    on_intervention()   an endpoint=intervention frame
    on_done()           endpoint=done: label -> buffer -> candidate fill -> updates -> ack

Each call holds the runtime's lock. The server serializes forward passes with its own
_PREDICT_LOCK (held around predict); on_done takes that lock too, always before the
runtime's own, so training never overlaps a rollout request and the two cannot deadlock.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

import flax.serialization
import jax
import numpy as np

from expo_ft.ultra.buffer import Episode, ReplayBuffer, Slot, Transition
from expo_ft.ultra.config import ExpoFtConfig
from expo_ft.ultra.episode import EpisodeRecorder, Observation
from expo_ft.ultra.learner import ExpoLearner, to_host
from expo_ft.ultra.normalize import ActionNormalizer
from expo_ft.ultra.rewards import HumanLabel, Outcome, SuccessDetector
from expo_ft.ultra.sampler import RtcCandidateSampler

LOGGER = logging.getLogger(__name__)

# Critic camera name -> openpi image slot (server_openpi.CAMERA_SLOTS values).
CAMERA_TO_SLOT = {"main": "base_0_rgb", "left_wrist": "left_wrist_0_rgb", "right_wrist": "right_wrist_0_rgb"}


def critic_images(images: dict[str, np.ndarray], cfg: ExpoFtConfig) -> np.ndarray:
    """openpi slot -> HWC uint8 dict to the critic's (V, S, S, 3), letterboxed exactly like
    the VLA's own ResizeImages transform."""
    from openpi_client import image_tools

    return np.stack([image_tools.resize_with_pad(images[CAMERA_TO_SLOT[k]], cfg.image_size, cfg.image_size)
                     for k in cfg.camera_keys]).astype(np.uint8)


def slot_images(critic: np.ndarray, cfg: ExpoFtConfig) -> dict[str, np.ndarray]:
    return {CAMERA_TO_SLOT[k]: critic[i] for i, k in enumerate(cfg.camera_keys)}


class JsonlLog:
    def __init__(self, path: Optional[Path]):
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, **fields) -> None:
        if self.path is None:
            return
        with self.path.open("a") as f:
            f.write(json.dumps({"kind": kind, "time": time.time(), **fields}) + "\n")


class ExpoRuntime:
    def __init__(self, policy: Any, cfg: ExpoFtConfig, *, norm_stats_dir: Path, run_dir: Optional[Path],
                 num_steps: int = 10, labeler: Optional[SuccessDetector] = None, use_quantiles: bool = True,
                 predict_lock: Optional[threading.Lock] = None, save_episodes: bool = True,
                 sampler: Any = None):
        self.cfg = cfg
        self.run_dir = None if run_dir is None else Path(run_dir)
        self.sampler = sampler or RtcCandidateSampler(policy, robot_action_dim=cfg.action_dim,
                                                      num_steps=num_steps, n=cfg.n_base, seed=cfg.seed)
        self.action_norm = ActionNormalizer.from_norm_stats_dir(norm_stats_dir, "actions", use_quantiles=use_quantiles)
        self.state_norm = ActionNormalizer.from_norm_stats_dir(norm_stats_dir, "state", use_quantiles=use_quantiles)
        self.learner = ExpoLearner(cfg)
        self.rng = jax.random.PRNGKey(cfg.seed + 1)
        self.state = self.learner.init(jax.random.PRNGKey(cfg.seed))
        self.buffer = ReplayBuffer(cfg, seed=cfg.seed)
        self.recorder = EpisodeRecorder(cfg)
        self.labeler = labeler or HumanLabel()
        self._lock = threading.Lock()
        self._predict_lock = predict_lock or threading.Lock()
        self.base_only = False  # execute base sample 0 (no selection/edits); still records and trains
        self._pending: Optional[dict] = None
        self._decisions: list[dict] = []
        self.episodes_done = 0
        self.save_episodes = save_episodes and self.run_dir is not None
        self.log = JsonlLog(None if self.run_dir is None else self.run_dir / "metrics.jsonl")
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1))

    # ------------------------------------------------------------------ helpers
    def observation(self, images: dict[str, np.ndarray], state_robot: np.ndarray) -> Observation:
        state_robot = np.asarray(state_robot, np.float32)
        return Observation(images=critic_images(images, self.cfg), state=self.state_norm.normalize(state_robot),
                           state_robot=state_robot)

    def _key(self):
        self.rng, key = jax.random.split(self.rng)
        return key

    # ------------------------------------------------------------------ rollout
    def predict(self, images: dict[str, np.ndarray], state_robot: np.ndarray, prompt: str, *,
                prefix: Optional[np.ndarray], prev_executed: Optional[int] = None) -> np.ndarray:
        """Robot-space (H, A) chunk for this observation; `prefix` is the robot-space (d, A)
        committed block from _resolve_prefix (None after a resume/intervention)."""
        with self._lock:
            t0 = time.perf_counter()
            obs = self.observation(images, state_robot)
            cands = self.sampler.sample(images, obs.state_robot, prompt, prefix)
            t1 = time.perf_counter()
            out = self.learner.select(self.state, obs.images, obs.state, cands.normalized,
                                      cands.prefix_len, self._key())
            qs = np.asarray(out["qs"])
            idx = 0 if self.base_only else int(out["idx"])
            chunk_norm = cands.normalized[0] if self.base_only else np.asarray(out["chunk"], np.float32)
            robot = self.sampler.unnormalize(chunk_norm[None], cands.model_state)[0]
            t2 = time.perf_counter()
            self._pending = dict(obs=obs, prefix_len=cands.prefix_len, prefix=prefix,
                                 candidates=cands.normalized, prev_executed=prev_executed, prompt=prompt)
            info = {"idx": idx, "edited": idx >= self.cfg.n_base, "q_max": float(qs.max()),
                    "q_mean": float(qs.mean()), "d": cands.prefix_len,
                    "sample_ms": 1e3 * (t1 - t0), "select_ms": 1e3 * (t2 - t1)}
            self._decisions.append(info)
            self.last_info = info
            return robot

    def record_sent(self, sent_robot: np.ndarray) -> None:
        """The chunk as it left the server (truncated/limited); recorded in normalized space."""
        with self._lock:
            p = self._pending
            assert p is not None, "record_sent without a preceding predict"
            self._pending = None
            self.recorder.on_policy(p["obs"], self.action_norm.normalize(sent_robot), prefix_len=p["prefix_len"],
                                    prefix_robot=p["prefix"], candidates=p["candidates"],
                                    prev_executed=p["prev_executed"], prompt=p["prompt"])

    def on_intervention(self, images: dict[str, np.ndarray], state_robot: np.ndarray,
                        human_rows_robot: np.ndarray, prev_executed: Optional[int] = None) -> None:
        with self._lock:
            if self.recorder.done:
                return
            self.recorder.on_intervention(self.observation(images, state_robot),
                                          self.action_norm.normalize(human_rows_robot), prev_executed)

    def connection_closed(self) -> None:
        with self._lock:
            if len(self.recorder) and not self.recorder.done:
                LOGGER.warning("connection closed mid-episode: discarding %d unlabelled events", len(self.recorder))
            self.recorder = EpisodeRecorder(self.cfg)
            self._pending = None
            self._decisions = []

    # ------------------------------------------------------------------ episode end
    def on_done(self, images: Optional[dict[str, np.ndarray]], state_robot: Optional[np.ndarray],
                prev_executed: Optional[int], done_reason: str = "") -> dict:
        """Label, store, fill candidates, train. Blocks (by design) until all of it is done."""
        with self._predict_lock, self._lock:
            rec, decisions = self.recorder, self._decisions
            self.recorder, self._decisions, self._pending = EpisodeRecorder(self.cfg), [], None
            if not len(rec):
                return {"stored": False, "reason": "empty episode"}
            obs = None if images is None else self.observation(images, state_robot)
            rec.on_done(obs, prev_executed)
            outcome = self.labeler.label(done_reason) or Outcome.ABORT
            ep = rec.build(outcome)
            summary = {"episode": self.episodes_done, "outcome": outcome.value, "stored": ep is not None,
                       "decisions": len(decisions),
                       "edit_selected": float(np.mean([d["edited"] for d in decisions])) if decisions else 0.0,
                       "q_max_mean": float(np.mean([d["q_max"] for d in decisions])) if decisions else 0.0,
                       "sample_ms": float(np.median([d["sample_ms"] for d in decisions])) if decisions else 0.0,
                       "select_ms": float(np.median([d["select_ms"] for d in decisions])) if decisions else 0.0}
            if ep is not None:
                summary.update(ep.info)
                self.buffer.add_episode(ep)
                if self.save_episodes:
                    save_episode(ep, self.run_dir / "episodes" / f"episode_{self.episodes_done:05d}.npz")
                t0 = time.perf_counter()
                summary["filled"] = self.fill_candidates()
                summary["fill_s"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                summary["train"] = self.train(self.cfg.updates_per_episode)
                summary["train_s"] = time.perf_counter() - t0
            self.episodes_done += 1
            summary.update(self.buffer.stats())
            self.log.write("episode", **summary)
            LOGGER.info("EXPO-FT episode %d: %s", summary["episode"],
                        {k: v for k, v in summary.items() if k != "train"})
            if self.run_dir is not None and self.episodes_done % 10 == 0:
                self.save(self.run_dir / "checkpoints" / f"state_{self.episodes_done:05d}.msgpack")
            return summary

    def fill_candidates(self) -> int:
        """Sample the N base chunks for every stored observation that has none (human
        anchors, abort-terminal observations, demos)."""
        slots = self.buffer.missing_candidates()
        c = self.cfg.exec_steps
        for i in slots:
            eid = int(self.buffer.episode_of[i])
            prompt = self.buffer.episodes[eid]["prompt"]
            d = int(self.buffer.prefix_len[i])
            prefix = self.buffer.prefix_robot[i, :d] if d else None
            cands = self.sampler.sample(slot_images(self.buffer.images[i], self.cfg),
                                        self.buffer.state_robot[i], prompt, prefix)
            self.buffer.set_candidates(np.array([i]), cands.normalized[None, :, :c])
        return len(slots)

    def train(self, n_updates: int) -> list[dict]:
        cfg = self.cfg
        if len(self.buffer) < cfg.batch_size:
            return []
        infos = []
        for _ in range(n_updates):
            batch = self.buffer.sample(cfg.batch_size * cfg.utd_ratio)
            self.state, info = self.learner.update(self.state, batch)
            infos.append(to_host(info))
        return infos

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(flax.serialization.to_bytes(jax.device_get(self.state)))
        LOGGER.info("saved EXPO-FT state to %s", path)

    def load(self, path: Path) -> None:
        self.state = flax.serialization.from_bytes(self.state, Path(path).read_bytes())

    def add_episodes(self, episodes: list[Episode]) -> None:
        """Seed the buffer (demos first) and fill their candidates."""
        with self._predict_lock, self._lock:
            for ep in episodes:
                self.buffer.add_episode(ep)
            n = self.fill_candidates()
            LOGGER.info("seeded %d episodes (%d transitions), filled %d candidate sets",
                        len(episodes), len(self.buffer), n)


# ---------------------------------------------------------------------- episode files

def save_episode(ep: Episode, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_base = next((s.candidates.shape[0] for s in ep.slots if s.candidates is not None), 0)
    c = ep.transitions[0].actions.shape[0]
    a = ep.transitions[0].actions.shape[1]
    has_c = np.array([s.candidates is not None for s in ep.slots])
    cands = np.zeros((len(ep.slots), n_base, c, a), np.float32)
    for i, s in enumerate(ep.slots):
        if s.candidates is not None:
            cands[i] = s.candidates
    max_d = max([s.prefix_len for s in ep.slots] + [1])
    prefix = np.zeros((len(ep.slots), max_d, a), np.float32)
    for i, s in enumerate(ep.slots):
        if s.prefix_len:
            prefix[i, : s.prefix_len] = s.prefix_robot[: s.prefix_len]
    np.savez_compressed(
        path,
        images=np.stack([s.images for s in ep.slots]),
        state=np.stack([s.state for s in ep.slots]),
        state_robot=np.stack([s.state_robot for s in ep.slots]),
        prefix_len=np.array([s.prefix_len for s in ep.slots]),
        prefix_robot=prefix, candidates=cands, has_candidates=has_c,
        t_src=np.array([t.src for t in ep.transitions]), t_next=np.array([t.next for t in ep.transitions]),
        t_actions=np.stack([t.actions for t in ep.transitions]),
        t_reward=np.array([t.reward for t in ep.transitions], np.float32),
        t_discount=np.array([t.discount for t in ep.transitions], np.float32),
        t_human=np.array([t.human_rows for t in ep.transitions]),
        meta=json.dumps({"prompt": ep.prompt, "is_demo": ep.is_demo, "outcome": ep.outcome, "info": ep.info}),
    )


def load_episode(path: Path) -> Episode:
    # Read each array once: NpzFile decompresses on every z[key], and per-slot views would
    # keep one full copy alive per slot.
    with np.load(path) as npz:
        z = {k: npz[k] for k in npz.files}
    meta = json.loads(str(z["meta"]))
    slots = [Slot(images=z["images"][i], state=z["state"][i], state_robot=z["state_robot"][i],
                  prefix_len=int(z["prefix_len"][i]),
                  prefix_robot=z["prefix_robot"][i, : int(z["prefix_len"][i])] if z["prefix_len"][i] else None,
                  candidates=z["candidates"][i] if z["has_candidates"][i] else None)
             for i in range(len(z["images"]))]
    transitions = [Transition(src=int(s), next=int(n), actions=a, reward=float(r), discount=float(d), human_rows=int(h))
                   for s, n, a, r, d, h in zip(z["t_src"], z["t_next"], z["t_actions"], z["t_reward"],
                                               z["t_discount"], z["t_human"])]
    return Episode(slots=slots, transitions=transitions, prompt=meta["prompt"], is_demo=meta["is_demo"],
                   outcome=meta["outcome"], info=meta["info"])
