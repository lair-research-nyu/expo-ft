#!/usr/bin/env python
"""EXPO-FT dry run: the full loop on recorded data, before anything touches the robot.

Starts the real server (policy-inference-spec server/server_openpi.py with --expo-ft) and
drives it over a local websocket with a simulated GEN2 client in "fixed-step
asynchronous" mode, replaying recorded LeRobot v3 episodes:

  * a request every C steps carrying the RTC prefix the robot would be executing
    (action_prefix = previous chunk past row C, prefix_change_start = simulated delay d);
  * scripted human takeovers: endpoint=intervention frames whose action rows are the
    recording's own actions, `prev_skipped_action_start` rows into a chunk, then a resume
    with no prefix;
  * endpoint=done with the terminal observation and a scripted label (the server runs
    with --expo-ft-label reason, so done_reason carries success/failure/abort), which
    makes the server store the episode, fill candidates and run the per-episode updates
    exactly as on the robot.

Observations come from the recording, not from executing our chunks (open loop), so this
tests plumbing, shapes, timing, learning dynamics on real images, and chunk seams -- not
policy quality. With --compare it runs a second pass with --expo-ft-base-only and prints
the seam ratio of both (edits on vs off), which is the offline check that edit
carry-over does not introduce jumps.

Usage (lair-openpi venv):
    PYTHONPATH=.:$HOME/projects/policy-inference-spec python scripts/expo_dry_run.py \\
        --checkpoint-path ~/projects/policy-inference-spec/models/openpi-ultra-cubestack-pi05-rtctrain-15000 \\
        --dataset ~/projects/ultra-tools/data/cubestack/lerobot --task "stack cubes" \\
        --episodes 6 --intervene-every 3 --compare
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

SPEC_ROOT = Path(os.environ.get("POLICY_INFERENCE_SPEC", "~/projects/policy-inference-spec")).expanduser()
EXPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPO_ROOT), str(SPEC_ROOT)]

from expo_ft.ultra.demos import DATASET_CAMERAS, LeRobotV3  # noqa: E402

WIRE_CAMERAS = {
    "base_0_rgb": "observation/images/main_image",
    "left_wrist_0_rgb": "observation/images/left_wrist_image",
    "right_wrist_0_rgb": "observation/images/right_wrist_image",
}
LABELS = ("success", "failure", "abort")


def wire_state_97(state25: np.ndarray) -> np.ndarray:
    """Inverse of server.wire._wire_state_25 (other entries zero)."""
    s = np.zeros(97, np.float32)
    s[0:8], s[32:40], s[64:70], s[88:91] = state25[0:8], state25[8:16], state25[16:22], state25[22:25]
    return s


class Episode:
    def __init__(self, reader: LeRobotV3, ep: int):
        self.ep = ep
        self.state, self.action = reader.arrays(ep)
        self.reader = reader
        self._frames: dict[int, dict] = {}

    def prefetch(self, steps):
        steps = sorted(set(int(s) for s in steps))
        cams = {slot: self.reader.frames(self.ep, key, steps) for key, slot in DATASET_CAMERAS.items()}
        for i, t in enumerate(steps):
            self._frames[t] = {slot: cams[slot][i] for slot in cams}

    def frame(self, t: int, task: str) -> dict:
        from policy_inference_spec.protocol import JOINT_STATE_KEY, MODEL_ID_KEY, SUBTASK_KEY, TASK_KEY

        f = {JOINT_STATE_KEY: wire_state_97(self.state[t]), TASK_KEY: task, SUBTASK_KEY: task,
             MODEL_ID_KEY: "expo-dry-run"}
        for slot, img in self._frames[t].items():
            f[WIRE_CAMERAS[slot]] = img
        return f


def plan_episode(t_len: int, c: int, takeover: tuple[int, int, int] | None):
    """Events (kind, step, extra) for one episode. takeover = (decision, rows_in, human_frames)."""
    events, t, k = [], 0, 0
    while t < t_len - 1:
        if takeover is not None and k == takeover[0]:
            j = min(takeover[1], t_len - 2 - t)
            events.append(("policy", t, {"resume": False}))
            t += j
            for _ in range(takeover[2]):
                if t + c >= t_len - 1:
                    break
                events.append(("human", t, {"first_rows": j}))
                j = None
                t += c
            takeover = None
            events.append(("policy", t, {"resume": True}))
            t += c
            k += 1
            continue
        events.append(("policy", t, {"resume": k == 0}))
        t += c
        k += 1
    return [e for e in events if e[1] < t_len - 1]


def seam_stats(timeline: np.ndarray, seams: list[int]) -> dict:
    """Mean |step| at the first freshly sampled row of each chunk vs. every other step."""
    deltas = np.abs(np.diff(timeline, axis=0))[:, :16].mean(axis=1)  # arms + grippers
    is_seam = np.zeros(len(deltas), bool)
    seams = [s - 1 for s in seams if 0 < s < len(timeline)]
    is_seam[seams] = True
    seam, intra = deltas[is_seam], deltas[~is_seam]
    return {"seam_mean": float(seam.mean()) if len(seam) else 0.0, "intra_mean": float(intra.mean()),
            "ratio": float(seam.mean() / intra.mean()) if len(seam) and intra.mean() > 0 else float("nan"),
            "seams": int(len(seam))}


async def drive(url: str, episodes: list[Episode], args) -> list[dict]:
    from policy_inference_spec.client import RemotePolicyClient

    c = args.exec_steps
    rng = np.random.default_rng(args.seed)
    results = []
    async with RemotePolicyClient(url) as client:
        for n, ep in enumerate(episodes):
            t_len = len(ep.action)
            takeover = None
            if args.intervene_every and (n + 1) % args.intervene_every == 0:
                takeover = (2, args.takeover_rows, args.takeover_frames)
            events = plan_episode(t_len, c, takeover)
            ep.prefetch([e[1] for e in events] + [t_len - 1])
            label = args.labels[n % len(args.labels)]
            prev_chunk = None
            timeline, seams, latencies, t_exec = [], [], [], 0
            for i, (kind, t, extra) in enumerate(events):
                nxt = events[i + 1][1] if i + 1 < len(events) else t_len - 1
                frame = ep.frame(t, args.task)
                if kind == "policy":
                    d = 0
                    if prev_chunk is not None and not extra["resume"]:
                        d = int(rng.integers(args.delay_min, args.delay_max + 1))
                        frame["action_prefix"] = prev_chunk[c:].astype(np.float32)
                        frame["prefix_change_start"] = d
                    t0 = time.perf_counter()
                    pred = await client.predict(frame)
                    latencies.append(time.perf_counter() - t0)
                    chunk = np.asarray(pred.actions_d, np.float32)
                    n_rows = nxt - t
                    timeline.append(chunk[:n_rows])
                    if d:
                        seams.append(t + d)
                    prev_chunk = chunk
                else:
                    rows = ep.action[t: t + c]
                    await client.record_human_intervention(
                        frame, action_chunk_hd=rows,
                        prev_skipped_action_start=extra["first_rows"])
                    timeline.append(rows[: nxt - t])
                    prev_chunk = None
                t_exec = nxt
            last_rows = max(1, (t_len - 1) - events[-1][1]) if events else 1
            last_rows = min(last_rows, 49)
            t0 = time.perf_counter()
            await client.mark_episode_done(label, final_wire_frame=ep.frame(t_len - 1, args.task),
                                           executed_steps=last_rows)
            done_s = time.perf_counter() - t0
            tl = np.concatenate(timeline)
            res = {"episode": n, "dataset_episode": ep.ep, "label": label, "steps": int(t_exec),
                   "decisions": sum(e[0] == "policy" for e in events), "takeover": takeover is not None,
                   "latency_ms_median": 1e3 * float(np.median(latencies)),
                   "latency_ms_max": 1e3 * float(np.max(latencies)), "done_s": done_s, **seam_stats(tl, seams)}
            print(json.dumps(res), flush=True)
            results.append(res)
    return results


def start_server(args, run_dir: Path, port: int, base_only: bool) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "server.server_openpi", "--checkpoint-path", str(args.checkpoint_path),
           "--rtc-train", "--task", args.task, "--port", str(port), "--expo-ft", "--expo-ft-label", "reason",
           "--expo-ft-run-dir", str(run_dir), "--expo-ft-set", f"exec_steps={args.exec_steps}", *args.set]
    # (all ExpoFtConfig overrides ride on the single --expo-ft-set above)
    if base_only:
        cmd.append("--expo-ft-base-only")
    if args.demos:
        cmd += ["--expo-ft-demos", str(args.dataset), "--expo-ft-num-demos", str(args.demos)]
    env = dict(os.environ, PYTHONPATH=f"{SPEC_ROOT}:{EXPO_ROOT}",
               XLA_PYTHON_CLIENT_MEM_FRACTION=str(args.mem_fraction))
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "server.log").open("w")
    proc = subprocess.Popen(cmd, cwd=SPEC_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    ready = threading.Event()

    def pump():
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if "Server listening on" in line:
                ready.set()
            if args.verbose or "EXPO-FT" in line or "Error" in line or "Traceback" in line:
                print("[server]", line.rstrip(), flush=True)

    threading.Thread(target=pump, daemon=True).start()
    while not ready.wait(2.0):
        if proc.poll() is not None:
            raise SystemExit(f"server exited with {proc.returncode}; see {run_dir / 'server.log'}")
    return proc


def run_pass(args, episodes, run_dir: Path, base_only: bool) -> list[dict]:
    proc = start_server(args, run_dir, args.port, base_only)
    try:
        return asyncio.run(drive(f"ws://127.0.0.1:{args.port}/ws", episodes, args))
    finally:
        proc.terminate()
        proc.wait(timeout=60)


def summarize(name: str, results: list[dict], run_dir: Path) -> None:
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()] \
        if (run_dir / "metrics.jsonl").exists() else []
    print(f"\n=== {name} ({run_dir}) ===")
    for r, m in zip(results, metrics):
        tr = m.get("train") or []
        last = tr[-1] if tr else {}
        print(f"ep {r['episode']:2d} {r['label']:8s} takeover={int(r['takeover'])} decisions={r['decisions']:3d} "
              f"lat={r['latency_ms_median']:.0f}/{r['latency_ms_max']:.0f}ms seam_ratio={r['ratio']:.2f} | "
              f"stored={m.get('stored')} trans={m.get('transitions')} human_rows={m.get('human_rows', 0)} "
              f"edit_sel={m.get('edit_selected', 0):.2f} critic_loss={last.get('critic_loss', float('nan')):.4f} "
              f"q={last.get('q', float('nan')):.3f} alpha={last.get('temperature', float('nan')):.3f} "
              f"train={m.get('train_s', 0):.1f}s fill={m.get('filled', 0)}")
    ratios = [r["ratio"] for r in results if np.isfinite(r["ratio"])]
    print(f"seam ratio (mean over episodes): {np.mean(ratios):.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-path", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True, help="LeRobot v3 root to replay.")
    p.add_argument("--task", default="stack cubes")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--max-len", type=int, default=900, help="Replay only episodes up to this many frames.")
    p.add_argument("--exec-steps", type=int, default=20, help="C = the client's steps per inference.")
    p.add_argument("--delay-min", type=int, default=4)
    p.add_argument("--delay-max", type=int, default=12, help="Simulated RTC delay d ~ U[min, max] rows.")
    p.add_argument("--intervene-every", type=int, default=3, help="Every n-th episode gets a takeover (0: never).")
    p.add_argument("--takeover-rows", type=int, default=7, help="Policy rows executed before the takeover.")
    p.add_argument("--takeover-frames", type=int, default=2, help="Intervention frames of C rows each.")
    p.add_argument("--labels", nargs="+", default=["success", "failure"], choices=LABELS,
                   help="Labels cycled over episodes.")
    p.add_argument("--demos", type=int, default=0, help="Seed this many demos from --dataset.")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="ExpoFtConfig overrides for the server, e.g. --set updates_per_episode=2")
    p.add_argument("--compare", action="store_true", help="Also run a base-only pass and compare seams.")
    p.add_argument("--run-dir", type=Path, default=Path("~/exp/expo-ft-dry-run").expanduser() / time.strftime("%Y_%m_%d_%H_%M_%S"))
    p.add_argument("--port", type=int, default=18190)
    p.add_argument("--mem-fraction", type=float, default=0.55)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    reader = LeRobotV3(args.dataset)
    ids = reader.episode_ids(task=args.task, exit_types=["success"])
    ids = [i for i in ids if 4 * args.exec_steps + 60 <= reader.episodes.loc[i, "length"] <= args.max_len]
    rng = np.random.default_rng(args.seed + 100)
    chosen = rng.choice(ids, size=min(args.episodes, len(ids)), replace=False)
    episodes = [Episode(reader, int(e)) for e in chosen]
    print(f"replaying dataset episodes {[e.ep for e in episodes]} "
          f"({sum(len(e.action) for e in episodes) / 50:.0f} s of recording)", flush=True)

    passes = [("expo-ft", False)] + ([("base-only", True)] if args.compare else [])
    for name, base_only in passes:
        run_dir = args.run_dir / name
        results = run_pass(args, episodes, run_dir, base_only)
        summarize(name, results, run_dir)


if __name__ == "__main__":
    main()
