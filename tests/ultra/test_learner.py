import jax
import jax.numpy as jnp
import numpy as np
import pytest

from expo_ft.ultra.config import CHEST, NECK
from expo_ft.ultra.learner import ExpoLearner, random_shift
from expo_ft.ultra.networks import tanh_gaussian_sample
from tests.ultra.conftest import replace, tiny_config


def _batch(cfg, n, key=0):
    rng = np.random.default_rng(key)
    v, s, a, c = len(cfg.camera_keys), cfg.image_size, cfg.action_dim, cfg.exec_steps
    img = lambda: rng.integers(0, 255, (n, v, s, s, 3), dtype=np.uint8)  # noqa: E731
    return {
        "images": img(),
        "state": rng.normal(size=(n, a)).astype(np.float32),
        "actions": rng.uniform(-1, 1, (n, c * a)).astype(np.float32),
        "base": rng.uniform(-1, 1, (n, c, a)).astype(np.float32),
        "prefix_len": rng.integers(0, cfg.max_prefix, n).astype(np.int32),
        "rewards": rng.uniform(0, 1, n).astype(np.float32),
        "discounts": np.full(n, cfg.decision_discount, np.float32),
        "next_images": img(),
        "next_state": rng.normal(size=(n, a)).astype(np.float32),
        "next_base": rng.uniform(-1, 1, (n, cfg.n_base, c, a)).astype(np.float32),
        "next_prefix_len": rng.integers(0, cfg.max_prefix, n).astype(np.int32),
    }


@pytest.fixture(scope="module")
def learner_and_state():
    cfg = tiny_config()
    learner = ExpoLearner(cfg)
    return learner, learner.init(jax.random.PRNGKey(0))


# --------------------------------------------------------------------- distribution

def test_tanh_gaussian_log_prob_matches_change_of_variables():
    mean = jnp.array([[0.3, -1.2, 0.0]])
    log_std = jnp.array([[-0.5, 0.2, -1.0]])
    u, lp = tanh_gaussian_sample(jax.random.PRNGKey(1), mean, log_std)
    x = jnp.arctanh(u)
    std = jnp.exp(log_std)
    direct = jnp.sum(-0.5 * ((x - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi) - jnp.log(1 - u**2), -1)
    np.testing.assert_allclose(lp, direct, rtol=1e-4)
    assert jnp.all(jnp.abs(u) < 1)


# --------------------------------------------------------------------- shapes

def test_network_shapes(learner_and_state):
    learner, st = learner_and_state
    cfg = learner.cfg
    b = _batch(cfg, 3)
    feats = learner._features(st.critic.params, b["images"], b["state"])
    assert feats.shape == (3, len(cfg.camera_keys) * cfg.image_latent_dim + cfg.state_latent_dim)
    qs = learner._qs(st.critic.params, feats, b["actions"])
    assert qs.shape == (cfg.num_qs, 3)
    mean, log_std = learner.edit_def.apply({"params": st.edit.params}, feats, b["actions"])
    assert mean.shape == log_std.shape == (3, cfg.edit_param_dim)


def test_select_shapes_and_candidate_structure(learner_and_state):
    learner, st = learner_and_state
    cfg = learner.cfg
    rng = np.random.default_rng(0)
    base = rng.uniform(-1, 1, (cfg.n_base, cfg.horizon, cfg.action_dim)).astype(np.float32)
    d = 6
    out = learner.select(st, _batch(cfg, 1)["images"][0], np.zeros(cfg.action_dim, np.float32),
                         base, d, jax.random.PRNGKey(3))
    cands = np.asarray(out["candidates"])
    assert out["chunk"].shape == (cfg.horizon, cfg.action_dim)
    assert out["qs"].shape == (cfg.n_candidates,)
    assert cands.shape == (cfg.n_candidates, cfg.horizon, cfg.action_dim)
    np.testing.assert_array_equal(cands[: cfg.n_base], base)
    np.testing.assert_allclose(np.asarray(out["chunk"]), cands[int(out["idx"])])
    off = cands[cfg.n_base:] - base[: cfg.n_edit]
    assert np.all(np.abs(off) <= cfg.edit_scale_vector + 1e-6)  # edit bounds
    assert np.all(off[:, :d] == 0)  # prefix pinned
    assert np.all(off[..., list(CHEST + NECK)] == 0)  # mask
    np.testing.assert_allclose(off[:, cfg.exec_steps:], np.broadcast_to(off[:, cfg.exec_steps - 1:cfg.exec_steps], off[:, cfg.exec_steps:].shape), atol=1e-6)


# --------------------------------------------------------------------- OTF argmax

def _fake_q(learner, target_flat):
    """Q(s, a) = -||a - target||^2 for every ensemble member: argmax is known."""
    target_flat = jnp.asarray(target_flat)

    def qs(params, feats, actions_flat):
        q = -jnp.sum((actions_flat - target_flat) ** 2, axis=-1)
        return jnp.broadcast_to(q, (learner.cfg.num_qs, q.shape[0]))

    return qs


def test_otf_argmax_picks_highest_q():
    cfg = tiny_config()
    learner = ExpoLearner(cfg)
    st = learner.init(jax.random.PRNGKey(0))
    rng = np.random.default_rng(1)
    base = rng.uniform(-1, 1, (cfg.n_base, cfg.horizon, cfg.action_dim)).astype(np.float32)
    good = 5
    learner._qs = _fake_q(learner, base[good, : cfg.exec_steps].reshape(-1) + 0.0)
    learner.select = jax.jit(learner._select)
    out = learner.select(st, _batch(cfg, 1)["images"][0], np.zeros(cfg.action_dim, np.float32), base, 0, jax.random.PRNGKey(0))
    qs = np.asarray(out["qs"])
    assert int(out["idx"]) == int(np.argmax(qs))
    # Base 5 is exactly the target, so no candidate can beat it unless an edit of 5 also scores 0.
    assert qs[int(out["idx"])] == pytest.approx(0.0, abs=1e-5)
    np.testing.assert_allclose(np.asarray(out["chunk"]), np.asarray(out["candidates"])[int(out["idx"])])


def test_otf_prefers_an_edit_when_it_is_better():
    cfg = tiny_config(n_base=2, n_edit=2)
    learner = ExpoLearner(cfg)
    st = learner.init(jax.random.PRNGKey(0))
    base = np.zeros((2, cfg.horizon, cfg.action_dim), np.float32)
    # Target sits slightly off every base sample in an editable dim: any non-zero edit
    # toward it wins, which the zero-mean edit policy produces with probability ~1/2 per dim.
    target = np.zeros((cfg.exec_steps, cfg.action_dim), np.float32)
    target[:, 0] = 0.05
    learner._qs = _fake_q(learner, target.reshape(-1))
    learner.select = jax.jit(learner._select)
    out = learner.select(st, _batch(cfg, 1)["images"][0], np.zeros(cfg.action_dim, np.float32), base, 0, jax.random.PRNGKey(7))
    qs = np.asarray(out["qs"])
    assert int(out["idx"]) == int(np.argmax(qs))
    if qs[2:].max() > qs[:2].max():
        assert int(out["idx"]) >= 2


# --------------------------------------------------------------------- TD target

def test_td_target_constant_q():
    cfg = tiny_config()
    learner = ExpoLearner(cfg)
    st = learner.init(jax.random.PRNGKey(0))
    learner._qs = lambda p, f, a: jnp.full((cfg.num_qs, a.shape[0]), 2.5)
    b = _batch(cfg, 5)
    b["discounts"][1] = 0.0  # terminal
    target, _ = learner._td_target(st, b, jax.random.PRNGKey(0))
    np.testing.assert_allclose(np.asarray(target), b["rewards"] + b["discounts"] * 2.5, rtol=1e-6)


def test_td_target_bootstraps_from_argmax_candidate_and_redq_min():
    cfg = tiny_config(n_edit=0)  # deterministic candidate set: just the cached base samples
    learner = ExpoLearner(cfg)
    st = learner.init(jax.random.PRNGKey(0))
    b = _batch(cfg, 3)
    flat = b["next_base"].reshape(3, cfg.n_base, -1)
    # Ensemble member i scores a = -||a||^2 - i: min over any pair of distinct members is
    # driven by the larger index, but the argmax over candidates is the same for all.
    def qs(params, feats, actions_flat):
        q = -jnp.sum(actions_flat**2, -1)
        return q[None] - jnp.arange(cfg.num_qs, dtype=q.dtype)[:, None]

    learner._qs = qs
    target, _ = learner._td_target(st, b, jax.random.PRNGKey(0))
    best = -np.min(np.sum(flat**2, -1), axis=1)  # max over candidates of -||a||^2
    # REDQ min over a random pair of 4 members: offset in {-1, -2, -3}.
    offs = (np.asarray(target) - b["rewards"]) / b["discounts"] - best
    assert np.all(np.isin(np.round(offs, 3), [-1.0, -2.0, -3.0]))


# --------------------------------------------------------------------- update

def test_update_runs_and_fits_rewards():
    cfg = tiny_config(utd_ratio=4, batch_size=8)
    learner = ExpoLearner(cfg)
    st = learner.init(jax.random.PRNGKey(0))
    b = _batch(cfg, 8)
    b["discounts"][:] = 0.0  # pure regression onto the rewards
    big = {k: np.concatenate([v] * cfg.utd_ratio) for k, v in b.items()}
    losses = []
    for _ in range(15):
        st, info = learner.update(st, big)
        losses.append(float(info["critic_loss"]))
    assert all(np.isfinite(losses))
    assert losses[-1] < 0.5 * losses[0]
    assert int(st.step) == 15
    assert set(info) >= {"critic_loss", "edit_loss", "entropy", "temperature", "target_q", "select_edit_ratio"}


def test_temperature_moves_toward_target_entropy():
    # alpha shrinks while entropy is above the target and grows while it is below.
    for target, direction in [(-1e4, -1), (1e4, +1)]:
        cfg = tiny_config(target_entropy=target)
        learner = ExpoLearner(cfg)
        st = learner.init(jax.random.PRNGKey(0))
        b = _batch(cfg, cfg.batch_size * cfg.utd_ratio)
        a0 = float(learner.temp_def.apply({"params": st.temp.params}))
        st, _ = learner.update(st, b)
        a1 = float(learner.temp_def.apply({"params": st.temp.params}))
        assert np.sign(a1 - a0) == direction


def test_random_shift_shape_and_content():
    imgs = np.arange(2 * 3 * 8 * 8 * 3, dtype=np.uint8).reshape(2, 3, 8, 8, 3)
    out = np.asarray(random_shift(jax.random.PRNGKey(0), imgs, 2))
    assert out.shape == imgs.shape and out.dtype == np.uint8
    np.testing.assert_array_equal(np.asarray(random_shift(jax.random.PRNGKey(0), imgs, 0)), imgs)


def test_target_network_tracks_online_slowly():
    cfg = replace(tiny_config(), tau_critic=0.5)
    learner = ExpoLearner(cfg)
    st = learner.init(jax.random.PRNGKey(0))
    b = _batch(cfg, cfg.batch_size * cfg.utd_ratio)
    st2, _ = learner.update(st, b)
    leaf = lambda t: jax.tree.leaves(t)[0]  # noqa: E731
    old, new, tgt = leaf(st.critic.params), leaf(st2.critic.params), leaf(st2.target_critic_params)
    assert not np.allclose(old, new)
    assert not np.allclose(tgt, old) and not np.allclose(tgt, new)


def test_update_compiles_once():
    cfg = tiny_config()
    learner = ExpoLearner(cfg)
    st = learner.init(jax.random.PRNGKey(0))
    b = _batch(cfg, cfg.batch_size * cfg.utd_ratio)
    for _ in range(3):
        st, _ = learner.update(st, b)
    assert learner.update._cache_size() == 1
