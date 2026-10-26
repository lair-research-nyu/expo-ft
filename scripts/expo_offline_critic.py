#!/usr/bin/env python
"""Offline critic check: is the EXPO-FT critic limited by data or by gradient steps?

Reloads the episodes a robot run saved (<run>/episodes/episode_*.npz), optionally seeds
the same demos the server would, and runs the server's own learner update (ExpoRuntime.
train: utd_ratio critic steps + one edit step per call) for many more steps than the
online run got. Every --eval-every updates it scores Q(s, a_executed) -- ensemble mean of
the online critic -- on every stored transition and compares it with the Monte-Carlo
return G_t = r_t + gamma^n_t * G_next:

    corr    Pearson correlation of Q with G over all transitions of a split
    mae     mean |Q - G|
    auc     P(Q at a decision of a success episode > Q at a decision of a failure one)
    q_succ / q_fail   mean Q over the decisions of success / failure episodes
    q0_succ / q0_fail mean Q at the first decision (the hardest call: G is tiny there)

on the online episodes it trains on, a held-out set of online episodes it never trains
on, and the demos. If the held-out numbers come up with enough steps, the online run is
compute-limited; if only the training split does, it is data-limited (memorizing).

Demos need the VLA (their base candidates are sampled with the RTC prefix pinned), so
--demos loads the checkpoint once, fills them, and caches them in <out>/demos/; later runs
with the same --out (or --demo-cache) skip the VLA. Stop the robot server first if the
GPU cannot hold a second copy of the VLA. Without --demos only the online episodes are
used and the VLA is never loaded (the checkpoint is still read for norm stats).

Usage (lair-openpi venv):
    PYTHONPATH=.:$HOME/projects/policy-inference-spec python scripts/expo_offline_critic.py \\
        --run ~/exp/expo-ft/2026_10_23_14_22_00 \\
        --checkpoint-path ~/projects/policy-inference-spec/models/openpi-ultra-cubestack-pi05-rtctrain-15000 \\
        --demos ~/projects/ultra-tools/data/towerstack-creo/lerobot/towerstack-creo --demo-task stack_rings \\
        --updates 500 --eval-every 25
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

SPEC_ROOT = Path(os.environ.get("POLICY_INFERENCE_SPEC", "~/projects/policy-inference-spec")).expanduser()
EXPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPO_ROOT), str(SPEC_ROOT)]

LOGGER = logging.getLogger("expo_offline_critic")


def mc_returns(ep) -> np.ndarray:
    """(len(transitions),) discounted return from each transition's source slot."""
    by_src = {t.src: t for t in ep.transitions}
    g: dict[int, float] = {}

    def ret(src: int) -> float:
        if src not in g:
            t = by_src[src]
            nxt = ret(t.next) if t.discount > 0 and t.next in by_src else 0.0
            g[src] = t.reward + t.discount * nxt
        return g[src]

    for t in sorted(ep.transitions, key=lambda t: -t.src):  # back to front: shallow recursion
        ret(t.src)
    return np.array([g[t.src] for t in ep.transitions], np.float32)


def split_holdout(eps: list, frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Indices (train, holdout): `frac` of each outcome, at least one of each when possible."""
    rng = np.random.default_rng(seed)
    hold: list[int] = []
    for outcome in sorted({e.outcome for e in eps}):
        idx = [i for i, e in enumerate(eps) if e.outcome == outcome]
        k = int(round(frac * len(idx)))
        if frac > 0 and len(idx) >= 2:
            k = max(k, 1)
        hold += rng.choice(idx, size=min(k, len(idx) - 1), replace=False).tolist() if k else []
    return [i for i in range(len(eps)) if i not in hold], sorted(hold)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) + 0.5 P(tie), by ranks."""
    if not len(pos) or not len(neg):
        return float("nan")
    x = np.concatenate([pos, neg])
    order = x.argsort(kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):  # average ranks over ties
        m = x == v
        ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class QScorer:
    """Ensemble-mean Q(s, a_executed) of the online critic, batched and jitted."""

    def __init__(self, learner, batch: int = 128):
        import jax

        self.batch = batch

        def q_fn(params, images, state, actions):
            feats = learner._features(params, images, state)  # noqa: SLF001
            return learner._qs(params, feats, actions).mean(axis=0)  # noqa: SLF001

        self._q = jax.jit(q_fn)

    def __call__(self, params, ep) -> np.ndarray:
        src = np.array([t.src for t in ep.transitions])
        images = np.stack([ep.slots[i].images for i in src])
        state = np.stack([ep.slots[i].state for i in src])
        actions = np.stack([t.actions.reshape(-1) for t in ep.transitions])
        out = []
        for s in range(0, len(src), self.batch):
            n = min(self.batch, len(src) - s)
            pad = self.batch - n  # fixed batch shape: one compile
            sl = slice(s, s + n)
            out.append(np.asarray(self._q(
                params, np.pad(images[sl], ((0, pad),) + ((0, 0),) * 4),
                np.pad(state[sl], ((0, pad), (0, 0))), np.pad(actions[sl], ((0, pad), (0, 0)))))[:n])
        return np.concatenate(out)


def split_metrics(qs: list[np.ndarray], gs: list[np.ndarray], eps: list) -> dict:
    if not eps:
        return {}
    q, g = np.concatenate(qs), np.concatenate(gs)
    succ = [i for i, e in enumerate(eps) if e.outcome == "success"]
    fail = [i for i, e in enumerate(eps) if e.outcome != "success"]
    cat = lambda ids: np.concatenate([qs[i] for i in ids]) if ids else np.zeros(0)  # noqa: E731
    first = lambda ids: np.array([qs[i][0] for i in ids]) if ids else np.zeros(0)  # noqa: E731
    mean = lambda x: float(x.mean()) if len(x) else float("nan")  # noqa: E731
    return {
        "n_eps": len(eps), "n": len(q),
        "corr": float(np.corrcoef(q, g)[0, 1]) if q.std() > 0 and g.std() > 0 else float("nan"),
        "mae": float(np.abs(q - g).mean()),
        "q_mean": float(q.mean()), "g_mean": float(g.mean()),
        "auc": auc(cat(succ), cat(fail)),
        "q_succ": mean(cat(succ)), "q_fail": mean(cat(fail)),
        "q0_succ": mean(first(succ)), "q0_fail": mean(first(fail)),
    }


def plot(path: Path, curves: list[dict], splits: dict, scores: dict, returns: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in ("train", "holdout", "demos") if splits.get(n)]
    fig, axes = plt.subplots(2, max(len(names), 2), figsize=(6 * max(len(names), 2), 8), squeeze=False)
    for j, name in enumerate(names):
        ax = axes[0, j]
        for i, ep in enumerate(splits[name]):
            color = "tab:green" if ep.outcome == "success" else "tab:red"
            ax.plot(scores[name][i], color=color, lw=1, alpha=0.8)
            ax.plot(returns[name][i], color=color, lw=1, ls="--", alpha=0.4)
        ax.set_title(f"{name}: Q (solid) vs MC return (dashed)")
        ax.set_xlabel("decision")
        ax.axhline(0, color="k", lw=0.5)
    steps = [c["grad_steps"] for c in curves]
    for j, (key, label) in enumerate([("corr", "corr(Q, G)"), ("auc", "success-vs-failure AUC")]):
        ax = axes[1, j]
        for name in names:
            ax.plot(steps, [c[name].get(key, np.nan) for c in curves], label=name, marker=".")
        ax.set_title(label)
        ax.set_xlabel("critic gradient steps")
        ax.legend()
    for j in range(2, axes.shape[1]):
        axes[1, j].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def load_demo_episodes(args, cfg, runtime_for_obs, cache: Path) -> list:
    from expo_ft.ultra.runtime import load_episode, save_episode, slot_images

    cached = sorted(cache.glob("demo_*.npz"))
    if cached:
        LOGGER.info("loading %d cached demos from %s", len(cached), cache)
        return [load_episode(p) for p in cached]
    from expo_ft.ultra.server_hook import load_demos

    ns = argparse.Namespace(expo_ft_demos=args.demos, expo_ft_demo_task=args.demo_task, task=args.demo_task,
                            expo_ft_demo_max_len=args.demo_max_len, expo_ft_num_demos=args.num_demos)
    demos = load_demos(ns, cfg, runtime_for_obs)
    c = cfg.exec_steps
    t0 = time.perf_counter()
    for k, ep in enumerate(demos):  # the same fill as ExpoRuntime.fill_candidates, per episode
        for s in ep.slots:
            if s.candidates is None:
                prefix = s.prefix_robot[: s.prefix_len] if s.prefix_len else None
                cands = runtime_for_obs.sampler.sample(slot_images(s.images, cfg), s.state_robot, ep.prompt, prefix)
                s.candidates = cands.normalized[:, :c]
        save_episode(ep, cache / f"demo_{k:03d}.npz")
        LOGGER.info("demo %d/%d filled (%d slots, %.0fs so far)", k + 1, len(demos), len(ep.slots),
                    time.perf_counter() - t0)
    return demos


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="A robot run dir (config.json + episodes/).")
    p.add_argument("--checkpoint-path", type=Path, required=True, help="The VLA checkpoint (norm stats; demo fill).")
    p.add_argument("--asset-id", default=None)
    p.add_argument("--demos", type=Path, default=None, help="LeRobot v3 root to seed demos from (needs the VLA once).")
    p.add_argument("--demo-task", default=None, help="Dataset task string of the demos, e.g. stack_rings.")
    p.add_argument("--num-demos", type=int, default=20)
    p.add_argument("--demo-max-len", type=int, default=1500)
    p.add_argument("--demo-cache", type=Path, default=None, help="Default: <out>/demos.")
    p.add_argument("--holdout-frac", type=float, default=0.25, help="Online episodes per outcome never trained on.")
    p.add_argument("--updates", type=int, default=500, help="learner.update calls (utd_ratio critic steps each).")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="ExpoFtConfig overrides (JSON values).")
    p.add_argument("--num-steps", type=int, default=10, help="VLA denoise steps for the demo fill.")
    p.add_argument("--out", type=Path, default=None, help="Default: <run>/offline_critic/<timestamp>.")
    p.add_argument("--mem-fraction", type=float, default=0.3)
    args = p.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(args.mem_fraction))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    assert args.demos is None or args.demo_task, "--demos needs --demo-task (the dataset's task string)"

    from expo_ft.ultra.config import ExpoFtConfig
    from expo_ft.ultra.runtime import ExpoRuntime, load_episode
    from server.server_openpi import _norm_stats_dir

    over = json.loads((args.run / "config.json").read_text())
    for kv in args.set:
        key, _, value = kv.partition("=")
        try:
            over[key] = json.loads(value)
        except json.JSONDecodeError:
            over[key] = value
    for k in ("camera_keys", "hidden_dims", "encoder_stage_sizes"):
        over[k] = tuple(over[k])
    cfg = ExpoFtConfig(**over)
    out = args.out or args.run / "offline_critic" / time.strftime("%Y_%m_%d_%H_%M_%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    norm_dir = _norm_stats_dir(args.checkpoint_path.expanduser(), args.asset_id)

    online = [load_episode(q) for q in sorted((args.run / "episodes").glob("episode_*.npz"))]
    assert online, f"no episodes in {args.run / 'episodes'}"
    train_idx, hold_idx = split_holdout(online, args.holdout_frac, cfg.seed)
    LOGGER.info("%d online episodes (%d success); holdout %s", len(online),
                sum(e.outcome == "success" for e in online), [(i, online[i].outcome) for i in hold_idx])

    demos = []
    if args.demos is not None:
        cache = args.demo_cache or out / "demos"
        if not sorted(cache.glob("demo_*.npz")):
            from server.server_openpi import OpenpiChunkPolicy

            policy = OpenpiChunkPolicy(args.checkpoint_path.expanduser(), rtc_train=True,
                                       num_steps=args.num_steps, asset_id=args.asset_id)
            vla_runtime = ExpoRuntime(policy._policy, cfg, norm_stats_dir=norm_dir, run_dir=None,  # noqa: SLF001
                                      num_steps=args.num_steps)
        else:
            vla_runtime = ExpoRuntime(None, cfg, norm_stats_dir=norm_dir, run_dir=None, sampler=object())
        demos = load_demo_episodes(args, cfg, vla_runtime, cache)
        del vla_runtime

    rt = ExpoRuntime(None, cfg, norm_stats_dir=norm_dir, run_dir=None, sampler=object())
    for ep in demos + [online[i] for i in train_idx]:  # demos first, as the server does
        rt.buffer.add_episode(ep)
    missing = rt.buffer.missing_candidates()
    assert not len(missing), f"{len(missing)} stored observations lack base candidates (need the VLA fill)"
    LOGGER.info("buffer: %s", rt.buffer.stats())

    splits = {"train": [online[i] for i in train_idx], "holdout": [online[i] for i in hold_idx], "demos": demos}
    returns = {n: [mc_returns(e) for e in eps] for n, eps in splits.items()}
    scorer = QScorer(rt.learner)
    log = (out / "metrics.jsonl").open("a")
    curves: list[dict] = []

    def evaluate(update: int, train_info: dict) -> dict:
        params = rt.state.critic.params
        scores = {n: [scorer(params, e) for e in eps] for n, eps in splits.items()}
        row = {"update": update, "grad_steps": update * cfg.utd_ratio, **{"train_info": train_info}}
        for n, eps in splits.items():
            row[n] = split_metrics(scores[n], returns[n], eps)
        log.write(json.dumps(row) + "\n")
        log.flush()
        curves.append(row)
        fmt = lambda m: " ".join(f"{k}={m[k]:.3f}" for k in ("corr", "auc", "q_succ", "q_fail", "q0_succ", "q0_fail"))  # noqa: E731
        LOGGER.info("update %d (%d critic steps) critic_loss=%.4f | %s", update, row["grad_steps"],
                    train_info.get("critic_loss", float("nan")),
                    " | ".join(f"{n}: {fmt(row[n])}" for n in splits if row[n]))
        plot(out / "critic.png", curves, splits, scores, returns)
        return row

    evaluate(0, {})
    t0 = time.perf_counter()
    for u in range(1, args.updates + 1):
        info = rt.train(1)[-1]
        if u % args.eval_every == 0 or u == args.updates:
            LOGGER.info("%.2fs per update call", (time.perf_counter() - t0) / u)
            evaluate(u, info)
    rt.save(out / "state.msgpack")
    LOGGER.info("done: %s (metrics.jsonl, critic.png, state.msgpack)", out)


if __name__ == "__main__":
    main()
