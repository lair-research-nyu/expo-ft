import numpy as np
import pytest

from expo_ft.ultra.buffer import ReplayBuffer
from expo_ft.ultra.episode import EpisodeRecorder, Observation
from expo_ft.ultra.normalize import ActionNormalizer
from expo_ft.ultra.rewards import Outcome
from tests.ultra.conftest import replace, tiny_config

G = 0.999


def _obs(cfg, tag):
    v, s = len(cfg.camera_keys), cfg.image_size
    return Observation(images=np.full((v, s, s, 3), tag % 255, np.uint8),
                       state=np.full(cfg.action_dim, tag, np.float32),
                       state_robot=np.full(cfg.action_dim, tag, np.float32))


def _chunk(cfg, tag):
    """Rows tagged tag*1000 + row index in every dim, so we can see where each row came from."""
    return (tag * 1000 + np.arange(cfg.horizon, dtype=np.float32))[:, None].repeat(cfg.action_dim, 1)


def _cands(cfg, tag):
    return np.full((cfg.n_base, cfg.horizon, cfg.action_dim), -tag, np.float32)


def _policy(rec, cfg, tag, d=0, prev=None):
    rec.on_policy(_obs(cfg, tag), _chunk(cfg, tag), prefix_len=d,
                  prefix_robot=np.zeros((d, cfg.action_dim), np.float32) if d else None,
                  candidates=_cands(cfg, tag), prev_executed=prev, prompt="stack")


def test_policy_only_episode():
    cfg = tiny_config()
    rec = EpisodeRecorder(cfg)
    _policy(rec, cfg, 1)
    _policy(rec, cfg, 2, d=5)
    _policy(rec, cfg, 3, d=7)
    rec.on_done(_obs(cfg, 9), prev_executed=7)
    ep = rec.build(Outcome.SUCCESS)
    assert len(ep.slots) == 4 and len(ep.transitions) == 3
    t0, t1, t2 = ep.transitions
    assert (t0.src, t0.next, t1.src, t1.next, t2.src, t2.next) == (0, 1, 1, 2, 2, 3)
    np.testing.assert_array_equal(t0.actions, _chunk(cfg, 1)[:20])
    np.testing.assert_array_equal(t1.actions, _chunk(cfg, 2)[:20])
    # 7 rows executed before done; the remaining 13 come from the planned chunk.
    np.testing.assert_array_equal(t2.actions, _chunk(cfg, 3)[:20])
    assert t0.reward == t1.reward == 0.0 and t2.reward == 1.0
    assert t0.discount == pytest.approx(G**20) and t2.discount == 0.0
    assert [s.prefix_len for s in ep.slots] == [0, 5, 7, 0]
    assert ep.slots[3].candidates is None and ep.slots[1].candidates.shape == (cfg.n_base, 20, 25)
    assert ep.info["steps"] == 47 and ep.info["human_rows"] == 0
    assert ep.prompt == "stack"


def test_takeover_splices_human_rows_into_the_chunk():
    cfg = tiny_config()
    rec = EpisodeRecorder(cfg)
    _policy(rec, cfg, 1)
    human = np.full((30, cfg.action_dim), 777.0, np.float32)
    rec.on_intervention(_obs(cfg, 2), human, prev_executed=8)  # took over 8 rows into chunk 1
    _policy(rec, cfg, 3)  # resume (d = 0) after all 30 human rows
    rec.on_done(_obs(cfg, 4), prev_executed=20)
    ep = rec.build(Outcome.SUCCESS)
    assert [t.src for t in ep.transitions] == [0, 1, 2]
    t0, t1, t2 = ep.transitions
    # 8 policy rows, then the human rows that replaced the rest of the chunk.
    np.testing.assert_array_equal(t0.actions[:8], _chunk(cfg, 1)[:8])
    np.testing.assert_array_equal(t0.actions[8:], human[:12])
    assert t0.next == 1 and t0.discount == pytest.approx(G**8) and t0.human_rows == 12
    # The human anchor's window is pure human; its next anchor is 30 rows later.
    np.testing.assert_array_equal(t1.actions, human[:20])
    assert t1.next == 2 and t1.discount == pytest.approx(G**30)
    assert ep.slots[1].candidates is None and ep.slots[1].prefix_len == 0
    np.testing.assert_array_equal(t2.actions, _chunk(cfg, 3)[:20])
    assert t2.reward == 1.0 and t2.discount == 0.0


def test_takeover_window_runs_into_the_resume_chunk():
    cfg = tiny_config()
    rec = EpisodeRecorder(cfg)
    _policy(rec, cfg, 1)
    rec.on_intervention(_obs(cfg, 2), np.full((6, 25), 5.0, np.float32), prev_executed=10)
    _policy(rec, cfg, 3)
    rec.on_done(None)
    ep = rec.build(Outcome.FAILURE)
    t0, t1, t2 = ep.transitions
    np.testing.assert_array_equal(t0.actions[10:16], 5.0)
    np.testing.assert_array_equal(t0.actions[16:], _chunk(cfg, 3)[:4])
    # Bootstrap at the latest anchor within C rows: the resume decision 16 rows in.
    assert t0.next == 2 and t0.discount == pytest.approx(G**16)
    assert t1.next == 2 and t1.discount == pytest.approx(G**6)
    assert t2.next == -1 and t2.reward == 0.0 and t2.discount == 0.0  # failure, no terminal obs


def test_dense_intervention_frames_are_subsampled_to_the_anchor_stride():
    cfg = tiny_config()
    rec = EpisodeRecorder(cfg)
    _policy(rec, cfg, 1)
    for k in range(12):  # 12 frames of 5 rows = 60 human rows
        rec.on_intervention(_obs(cfg, 10 + k), np.full((5, 25), k, np.float32), prev_executed=4 if k == 0 else None)
    _policy(rec, cfg, 3)
    rec.on_done(_obs(cfg, 4), prev_executed=20)
    ep = rec.build(Outcome.SUCCESS)
    steps = [0, 4, 24, 44, 64, 84]  # policy, takeover, every 20 human rows, resume, terminal
    assert len(ep.slots) == len(steps)
    assert [t.discount for t in ep.transitions[:4]] == pytest.approx([G**4, G**20, G**20, G**20])
    assert ep.info["human_rows"] == 60


def test_abort_truncates_or_discards():
    cfg = tiny_config()
    rec = EpisodeRecorder(cfg)
    _policy(rec, cfg, 1)
    _policy(rec, cfg, 2, d=4)
    rec.on_done(_obs(cfg, 3), prev_executed=11)
    ep = rec.build(Outcome.ABORT)
    last = ep.transitions[-1]
    assert last.reward == 0.0 and last.next == 2 and last.discount == pytest.approx(G**11)  # bootstraps
    assert rec.build(Outcome.ABORT) is not None
    assert EpisodeRecorder(replace(cfg, abort_mode="discard")).build(Outcome.ABORT) is None

    rec2 = EpisodeRecorder(cfg)
    _policy(rec2, cfg, 1)
    _policy(rec2, cfg, 2)
    rec2.on_done(None)
    ep2 = rec2.build(Outcome.ABORT)
    assert len(ep2.transitions) == 1  # the last one has nothing to bootstrap from: dropped


def test_done_with_zero_rows_since_the_last_decision():
    cfg = tiny_config()
    rec = EpisodeRecorder(cfg)
    _policy(rec, cfg, 1)
    _policy(rec, cfg, 2)
    rec.on_done(_obs(cfg, 3), prev_executed=0)
    ep = rec.build(Outcome.SUCCESS)
    assert len(ep.slots) == 2 and ep.transitions[-1].next == -1 and ep.transitions[-1].reward == 1.0


# --------------------------------------------------------------------- buffer

def _episode(cfg, tags, outcome=Outcome.SUCCESS, human=False):
    rec = EpisodeRecorder(cfg)
    for i, t in enumerate(tags):
        _policy(rec, cfg, t, d=3 if i else 0)
        if human and i == 0:
            rec.on_intervention(_obs(cfg, t + 50), np.zeros((20, 25), np.float32), prev_executed=10)
    rec.on_done(_obs(cfg, tags[-1] + 100), prev_executed=20)
    return rec.build(outcome)


def test_buffer_sample_links_next_observations():
    cfg = tiny_config(n_base=2)
    buf = ReplayBuffer(cfg, capacity=50)
    buf.add_episode(_episode(cfg, [1, 2, 3]))
    assert len(buf) == 3
    b = buf.sample(64)
    n = 64
    assert b["images"].shape == (n, 3, 32, 32, 3) and b["actions"].shape == (n, 20 * 25)
    assert b["next_base"].shape == (n, 2, 20, 25) and b["base"].shape == (n, 20, 25)
    # Each observation's state is its tag; the next observation's is the next tag (or terminal).
    tag = b["state"][:, 0]
    nxt = b["next_state"][:, 0]
    expected = {1: 2, 2: 3, 3: 103}  # the last one points at the terminal observation
    assert all(expected[int(a)] == int(c) for a, c in zip(tag, nxt))
    assert np.all(b["discounts"][tag == 3] == 0) and np.all(b["rewards"][tag == 3] == 1)
    # Base and next_base come from the cached candidates (-tag).
    assert np.all(b["base"][:, 0, 0] == -tag)
    np.testing.assert_array_equal(b["next_prefix_len"][tag == 1], 3)


def test_buffer_missing_candidates_and_fill():
    cfg = tiny_config(n_base=2)
    buf = ReplayBuffer(cfg, capacity=50)
    buf.add_episode(_episode(cfg, [1, 2], human=True))
    missing = buf.missing_candidates()
    assert len(missing) == 1 and buf.state[missing[0], 0] == 51  # the takeover anchor
    with pytest.raises(AssertionError):
        for _ in range(20):
            buf.sample(16)
    buf.set_candidates(missing, np.zeros((1, 2, 20, 25), np.float32))
    assert len(buf.missing_candidates()) == 0
    buf.sample(16)


def test_buffer_fifo_eviction_keeps_demos():
    cfg = tiny_config(n_base=2)
    buf = ReplayBuffer(cfg, capacity=12)
    demo = _episode(cfg, [1, 2])
    demo.is_demo = True
    buf.add_episode(demo)  # 3 slots
    ids = [buf.add_episode(_episode(cfg, [10 * k + 1, 10 * k + 2])) for k in range(1, 6)]
    stats = buf.stats()
    assert stats["episodes_demo"] == 1
    assert stats["episodes_online"] == 3  # 9 online slots fit 3 episodes of 3
    assert set(buf.episodes) == {0, *ids[-3:]}
    assert set(np.unique(buf.state[buf.has_transition, 0]).astype(int)) >= {1, 2}
    buf.sample(8)


def test_normalizer_roundtrip_and_formula():
    stats = {"q01": [-1.0, 0.0], "q99": [1.0, 4.0]}
    n = ActionNormalizer(stats)
    x = np.array([[0.0, 1.0], [1.0, 4.0]], np.float32)
    np.testing.assert_allclose(n.normalize(x), [[0.0, -0.5], [1.0, 1.0]], atol=1e-5)
    np.testing.assert_allclose(n.unnormalize(n.normalize(x)), x, atol=1e-5)
