"""ExpoRuntime end to end with a fake VLA sampler (no checkpoint needed)."""

import json

import numpy as np

from expo_ft.ultra.normalize import ActionNormalizer
from expo_ft.ultra.rewards import FixedLabel, Outcome
from expo_ft.ultra.runtime import CAMERA_TO_SLOT, ExpoRuntime, load_episode
from expo_ft.ultra.sampler import Candidates
from tests.ultra.conftest import tiny_config


class FakeSampler:
    """Pinned-prefix base chunks around the current state, like the VLA would roughly give."""

    def __init__(self, cfg, norm):
        self.cfg, self.norm, self.rng, self.calls = cfg, norm, np.random.default_rng(0), []

    def sample(self, images, state_robot, prompt, prefix_robot=None, noise=None):
        cfg = self.cfg
        self.calls.append(0 if prefix_robot is None else len(prefix_robot))
        z = self.rng.normal(0, 0.05, (cfg.n_base, cfg.horizon, cfg.action_dim)).cumsum(1).astype(np.float32)
        d = 0 if prefix_robot is None else len(prefix_robot)
        pre = self.norm.normalize(prefix_robot) if d else np.zeros((0, cfg.action_dim), np.float32)
        z[:, :d] = pre
        return Candidates(normalized=z, robot=self.norm.unnormalize(z), state=np.zeros(cfg.action_dim, np.float32),
                          prefix_len=d, prefix=pre, model_state=np.zeros(32, np.float32))

    def unnormalize(self, chunks, state):
        return self.norm.unnormalize(chunks)


def _runtime(tmp_path, **cfg_kw):
    cfg = tiny_config(**cfg_kw)
    stats = {"q01": [-1.0] * 25, "q99": [1.0] * 25}
    (tmp_path / "norm_stats.json").write_text(json.dumps({"norm_stats": {"actions": stats, "state": stats}}))
    norm = ActionNormalizer(stats)
    return ExpoRuntime(None, cfg, norm_stats_dir=tmp_path, run_dir=tmp_path / "run",
                       labeler=FixedLabel(by_reason={o.value: o for o in Outcome}),
                       sampler=FakeSampler(cfg, norm))


def _images(v=0):
    return {slot: np.full((360, 640, 3), v, np.uint8) for slot in CAMERA_TO_SLOT.values()}


def _run_episode(rt, n_decisions=5, takeover_at=None, label="success", d=6):
    cfg = rt.cfg
    prev = None
    for k in range(n_decisions):
        prefix = None if (prev is None or k == takeover_at) else prev[cfg.exec_steps: cfg.exec_steps + d]
        chunk = rt.predict(_images(k), np.zeros(25, np.float32), "stack", prefix=prefix)
        assert chunk.shape == (cfg.horizon, 25)
        if prefix is not None:
            np.testing.assert_allclose(chunk[:d], prefix, atol=1e-5)  # pinned through selection+edit
        rt.record_sent(chunk)
        prev = chunk
        if takeover_at is not None and k == takeover_at - 1:
            rt.on_intervention(_images(99), np.zeros(25, np.float32), np.zeros((cfg.exec_steps, 25), np.float32),
                               prev_executed=8)
    return rt.on_done(_images(50), np.zeros(25, np.float32), prev_executed=11, done_reason=label)


def test_runtime_episode_loop(tmp_path):
    rt = _runtime(tmp_path, updates_per_episode=2)
    s1 = _run_episode(rt, takeover_at=3)
    assert s1["stored"] and s1["outcome"] == "success" and s1["human_rows"] == 20
    assert s1["filled"] == 1  # the takeover anchor got its base candidates sampled
    assert len(s1["train"]) == 0 or len(s1["train"]) == 2
    s2 = _run_episode(rt, label="failure")
    assert s2["stored"] and len(s2["train"]) == 2
    assert all(np.isfinite(t["critic_loss"]) for t in s2["train"])
    assert rt.buffer.stats()["episodes_online"] == 2
    log = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert [e["outcome"] for e in log] == ["success", "failure"]
    # Episodes persist and reload identically.
    files = sorted((tmp_path / "run" / "episodes").glob("*.npz"))
    assert len(files) == 2
    ep = load_episode(files[0])
    assert len(ep.transitions) == len(ep.slots) - 1 and ep.outcome == "success"


def test_abort_is_truncated_and_bootstraps(tmp_path):
    rt = _runtime(tmp_path)
    s = _run_episode(rt, n_decisions=3, label="abort")
    assert s["stored"] and s["filled"] == 1  # terminal observation needs candidates to bootstrap from
    last = rt.buffer.has_transition & (rt.buffer.discounts > 0) & (rt.buffer.rewards == 0)
    assert last.sum() == 3


def test_disconnect_discards_unlabelled_episode(tmp_path):
    rt = _runtime(tmp_path)
    rt.record_sent(rt.predict(_images(), np.zeros(25, np.float32), "stack", prefix=None))
    rt.connection_closed()
    assert rt.on_done(None, None, None)["stored"] is False
    assert len(rt.buffer) == 0


def test_base_only_executes_first_base_sample(tmp_path):
    rt = _runtime(tmp_path)
    rt.base_only = True
    chunk = rt.predict(_images(), np.zeros(25, np.float32), "stack", prefix=None)
    assert rt.last_info["idx"] == 0 and not rt.last_info["edited"]
    assert chunk.shape == (rt.cfg.horizon, 25)


def test_load_episode_holds_one_copy_of_each_array(tmp_path):
    rt = _runtime(tmp_path)
    _run_episode(rt, n_decisions=4)
    ep = load_episode(sorted((tmp_path / "run" / "episodes").glob("*.npz"))[0])
    # Per-slot arrays are views of a single loaded array, not of one decompressed copy each.
    assert len({id(s.images.base) for s in ep.slots}) == 1
