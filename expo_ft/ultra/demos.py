"""Seed the replay buffer with LeRobot v3 demonstrations (paper Sec. 4: the demos the VLA
was SFT'd on initialize the buffer).

A minimal v3 reader (parquet + PyAV) because the lerobot package in the lair-openpi venv
predates v3. Each demo becomes decision windows at stride C: an observation every C
frames, the next C recorded actions as the chunk, and an RTC delay d drawn from the
online distribution (uniform over the trained delays [0, max_prefix), d = 0 on the first
window, as on the robot) whose first d rows act as the committed prefix. The demos go
through the same EpisodeRecorder as online episodes; their base candidates are sampled
afterwards (ExpoRuntime.fill_candidates) with that prefix pinned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from expo_ft.ultra.buffer import Episode
from expo_ft.ultra.config import ExpoFtConfig
from expo_ft.ultra.episode import EpisodeRecorder, Observation
from expo_ft.ultra.rewards import Outcome

# Dataset camera feature -> openpi image slot (see server_openpi.CAMERA_SLOTS / DUMP_CAMERA_NAMES).
DATASET_CAMERAS = {
    "observation.images.head": "base_0_rgb",
    "observation.images.left_wrist": "left_wrist_0_rgb",
    "observation.images.right_wrist": "right_wrist_0_rgb",
}


class LeRobotV3:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.fps = float(self.info["fps"])
        files = sorted((self.root / "meta" / "episodes").glob("*/*.parquet"))
        cols = None
        self.episodes = pd.concat([pd.read_parquet(f, columns=cols) for f in files]).set_index("episode_index")
        self.episodes = self.episodes[[c for c in self.episodes.columns if not c.startswith("stats/")]]

    def episode_ids(self, task: Optional[str] = None, exit_types: Optional[Sequence[str]] = None) -> list[int]:
        df = self.episodes
        if task is not None:
            df = df[df["tasks"].apply(lambda t: task in list(t))]
        if exit_types is not None and "exit_type" in df.columns:
            df = df[df["exit_type"].isin(list(exit_types))]
        return [int(i) for i in df.index]

    def task(self, ep: int) -> str:
        return str(list(self.episodes.loc[ep, "tasks"])[0])

    def arrays(self, ep: int) -> tuple[np.ndarray, np.ndarray]:
        """(state (T, A), action (T, A)) in robot space."""
        row = self.episodes.loc[ep]
        path = self.root / self.info["data_path"].format(chunk_index=int(row["data/chunk_index"]),
                                                         file_index=int(row["data/file_index"]))
        df = pd.read_parquet(path, columns=["index", "observation.state", "action"])
        df = df[(df["index"] >= row["dataset_from_index"]) & (df["index"] < row["dataset_to_index"])].sort_values("index")
        return np.stack(df["observation.state"].to_numpy()).astype(np.float32), \
            np.stack(df["action"].to_numpy()).astype(np.float32)

    def frames(self, ep: int, key: str, frame_indices: Sequence[int]) -> list[np.ndarray]:
        """Decode the given frames (episode-relative) of one camera as HWC uint8."""
        import av

        row = self.episodes.loc[ep]
        path = self.root / self.info["video_path"].format(
            video_key=key, chunk_index=int(row[f"videos/{key}/chunk_index"]),
            file_index=int(row[f"videos/{key}/file_index"]))
        t0 = float(row[f"videos/{key}/from_timestamp"])
        wanted = sorted(set(int(i) for i in frame_indices))
        out: dict[int, np.ndarray] = {}
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            container.seek(int(t0 / stream.time_base), stream=stream, backward=True, any_frame=False)
            j = 0
            for frame in container.decode(stream):
                idx = int(round((float(frame.pts * stream.time_base) - t0) * self.fps))
                while j < len(wanted) and wanted[j] < idx:
                    j += 1  # a frame the decoder skipped: take the next one that exists
                if j < len(wanted) and idx == wanted[j]:
                    out[wanted[j]] = frame.to_ndarray(format="rgb24")
                    j += 1
                if j >= len(wanted):
                    break
        missing = [i for i in wanted if i not in out]
        assert not missing, f"episode {ep} {key}: could not decode frames {missing[:5]}"
        return [out[int(i)] for i in frame_indices]


def demo_episode(reader: LeRobotV3, ep: int, cfg: ExpoFtConfig, *, observe: Callable[[dict, np.ndarray], Observation],
                 normalize_actions: Callable[[np.ndarray], np.ndarray], rng: np.random.Generator,
                 prompt: Optional[str] = None) -> Episode:
    """One demonstration as decision windows. `observe` is ExpoRuntime.observation."""
    state, action = reader.arrays(ep)
    t_len = len(action)
    c, h = cfg.exec_steps, cfg.horizon
    starts = list(range(0, t_len - 1, c))
    last = t_len - 1
    frames_needed = starts + [last]
    cams = {slot: reader.frames(ep, key, frames_needed) for key, slot in DATASET_CAMERAS.items()}
    frame_pos = {f: i for i, f in enumerate(frames_needed)}
    rec = EpisodeRecorder(cfg)
    norm_actions = normalize_actions(action)
    for k, t in enumerate(starts):
        d = 0 if k == 0 else min(int(rng.integers(0, cfg.max_prefix)), t_len - t)
        rows = norm_actions[t: t + h]
        if len(rows) < h:
            rows = np.concatenate([rows, np.repeat(rows[-1:], h - len(rows), 0)])
        obs = observe({slot: cams[slot][frame_pos[t]] for slot in cams}, state[t])
        rec.on_policy(obs, rows, prefix_len=d, prefix_robot=action[t: t + d] if d else None,
                      candidates=None, prev_executed=None if k == 0 else c, prompt=prompt or reader.task(ep))
    final_obs = observe({slot: cams[slot][frame_pos[last]] for slot in cams}, state[last])
    rec.on_done(final_obs, prev_executed=last - starts[-1])
    out = rec.build(Outcome.SUCCESS)
    out.is_demo = True
    out.info["source"] = f"{reader.root}#{ep}"
    return out
