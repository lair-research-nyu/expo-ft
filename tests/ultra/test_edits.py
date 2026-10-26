import jax
import jax.numpy as jnp
import numpy as np
import pytest

from expo_ft.ultra.config import CHEST, LEFT_GRIPPER, NECK, RIGHT_ARM, ExpoFtConfig
from expo_ft.ultra.edits import EditExpander, interpolation_matrix, knot_rows, row_envelope
from tests.ultra.conftest import tiny_config


def test_config_geometry():
    cfg = ExpoFtConfig()
    assert cfg.edit_dims == tuple(range(16))  # both arms + grippers, chest/neck masked
    assert cfg.num_knots == 5 and cfg.edit_param_dim == 80
    assert cfg.chunk_flat_dim == 20 * 25
    assert cfg.decision_discount == pytest.approx(0.999**20)
    s = cfg.edit_scale_vector
    np.testing.assert_allclose(s[list(RIGHT_ARM)], 0.1)
    assert s[LEFT_GRIPPER[0]] == pytest.approx(0.2)
    assert np.all(s[list(CHEST + NECK)] == 0)
    assert ExpoFtConfig(edit_knot_stride=1).edit_param_dim == 20 * 16


def test_config_rejects_cache_with_vla_training():
    with pytest.raises(AssertionError):
        ExpoFtConfig(train_vla=True, cache_base_candidates=True)
    ExpoFtConfig(train_vla=True, cache_base_candidates=False)


@pytest.mark.parametrize("stride", [1, 3, 5, 20])
def test_interpolation_matrix(stride):
    c, h = 20, 50
    cfg = ExpoFtConfig(edit_knot_stride=stride)
    k = cfg.num_knots
    w = interpolation_matrix(c, k, h)
    np.testing.assert_allclose(w.sum(1), 1.0, atol=1e-6)
    assert (w >= 0).all()
    # Each knot is reproduced exactly at its own row.
    for j, r in enumerate(knot_rows(c, k)):
        if float(r).is_integer():
            assert w[int(r), j] == pytest.approx(1.0)
    # Hold: every row from C-1 on copies the last knot.
    np.testing.assert_allclose(w[c - 1:], np.eye(k)[[k - 1] * (h - c + 1)])
    if stride == 1:
        np.testing.assert_allclose(w[:c], np.eye(c))


def test_envelope():
    env = np.asarray(row_envelope(3, 10, ramp=4))
    np.testing.assert_allclose(env, [0, 0, 0, .25, .5, .75, 1, 1, 1, 1])
    np.testing.assert_allclose(np.asarray(row_envelope(0, 4, ramp=0)), [1, 1, 1, 1])
    np.testing.assert_allclose(np.asarray(row_envelope(jnp.array([0, 2]), 3, ramp=0)), [[1, 1, 1], [0, 0, 1]])


def _random_u(cfg, n, key=0, extreme=False):
    u = jax.random.uniform(jax.random.PRNGKey(key), (n, cfg.edit_param_dim), minval=-1, maxval=1)
    return jnp.sign(u) if extreme else u


def test_edit_bounds_mask_prefix_and_carry_over():
    cfg = tiny_config(edit_ramp_rows=5)
    ex = EditExpander(cfg)
    c, h = cfg.exec_steps, cfg.horizon
    d = jnp.array([0, 4, 14])
    off = np.asarray(ex(_random_u(cfg, 3, extreme=True), d, h))
    assert off.shape == (3, h, cfg.action_dim)
    scale = cfg.edit_scale_vector
    # Bounds: never more than the per-dim scale (|u| <= 1, convex interpolation, envelope <= 1).
    assert np.all(np.abs(off) <= scale + 1e-6)
    # Mask: chest and neck untouched.
    assert np.all(off[..., list(CHEST + NECK)] == 0)
    # Prefix rows untouched.
    for i, di in enumerate(np.asarray(d)):
        assert np.all(off[i, :di] == 0)
    # Carry-over: edited - base is constant on rows >= C and equals row C-1.
    np.testing.assert_allclose(off[:, c:], np.broadcast_to(off[:, c - 1:c], off[:, c:].shape), atol=1e-7)
    # With |u| = 1 on the last knot and the ramp finished, row C-1 sits at the full scale.
    np.testing.assert_allclose(np.abs(off[:, c - 1, list(cfg.edit_dims)]), np.broadcast_to(scale[list(cfg.edit_dims)], (3, 16)), atol=1e-6)


def test_edit_stride_one_is_per_row():
    cfg = tiny_config(edit_knot_stride=1, edit_ramp_rows=0)
    ex = EditExpander(cfg)
    u = _random_u(cfg, 1)
    off = np.asarray(ex(u, jnp.array([0]), cfg.exec_steps))[0]
    per_row = np.asarray(u).reshape(cfg.exec_steps, len(cfg.edit_dims)) * cfg.edit_scale_vector[list(cfg.edit_dims)]
    np.testing.assert_allclose(off[:, list(cfg.edit_dims)], per_row, atol=1e-6)


def test_log_scale_sum():
    cfg = ExpoFtConfig()
    expected = 5 * (14 * np.log(0.1) + 2 * np.log(0.2))
    assert EditExpander(cfg).log_scale_sum() == pytest.approx(expected, rel=1e-6)
