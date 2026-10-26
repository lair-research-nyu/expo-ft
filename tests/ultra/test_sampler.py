"""RtcCandidateSampler against the real checkpoint (skipped when it is not on disk).

Run with policy-inference-spec on the path:
    PYTHONPATH=.:$HOME/projects/policy-inference-spec pytest tests/ultra/test_sampler.py
"""

import os
import time
from pathlib import Path

import numpy as np
import pytest

CKPT = Path(os.environ.get(
    "EXPO_TEST_CHECKPOINT",
    "~/projects/policy-inference-spec/models/openpi-ultra-cubestack-pi05-rtctrain-15000")).expanduser()
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason=f"no checkpoint at {CKPT}")


@pytest.fixture(scope="module")
def served():
    from server.server_openpi import CAMERA_SLOTS, OpenpiChunkPolicy

    from expo_ft.ultra.sampler import RtcCandidateSampler

    policy = OpenpiChunkPolicy(CKPT, rtc_train=True)
    sampler = RtcCandidateSampler(policy._policy, robot_action_dim=25, num_steps=10, n=4)
    rng = np.random.default_rng(0)
    images = {slot: rng.integers(0, 255, (360, 640, 3), dtype=np.uint8) for slot in CAMERA_SLOTS.values()}
    # A plausible robot-space state/prefix: the middle of the action quantile range.
    from expo_ft.ultra.normalize import ActionNormalizer
    from server.server_openpi import _norm_stats_dir

    norm = ActionNormalizer.from_norm_stats_dir(_norm_stats_dir(CKPT, None), "actions")
    state = norm.unnormalize(np.zeros(25, np.float32))
    prefix = norm.unnormalize(rng.uniform(-0.5, 0.5, (6, 25)).astype(np.float32))
    return policy, sampler, images, state, prefix, norm


def test_matches_policy_infer_for_the_same_noise(served):
    policy, sampler, images, state, prefix, _ = served
    noise = np.random.default_rng(1).normal(size=(4, sampler.horizon, sampler.model_action_dim)).astype(np.float32)
    cands = sampler.sample(images, state, "stack the cubes", prefix, noise=noise)
    assert cands.normalized.shape == cands.robot.shape == (4, 50, 25)
    for i in range(4):
        obs = {"state": state, "image": images, "image_mask": {k: np.True_ for k in images},
               "prompt": "stack the cubes", "actions": np.pad(prefix, ((0, 50 - 6), (0, 0))), "prefix_len": 6}
        ref = np.asarray(policy._policy.infer(obs, noise=noise[i])["actions"])[:, :25]
        # bf16 weights, batch 1 vs batch N: allow a little numerical drift.
        np.testing.assert_allclose(cands.robot[i], ref, atol=3e-2)


def test_prefix_is_pinned_and_normalizer_agrees(served):
    _, sampler, images, state, prefix, norm = served
    cands = sampler.sample(images, state, "stack the cubes", prefix)
    assert cands.prefix_len == 6
    np.testing.assert_allclose(cands.normalized[:, :6], np.broadcast_to(cands.prefix, (4, 6, 25)), atol=1e-6)
    np.testing.assert_allclose(cands.robot[:, :6], np.broadcast_to(prefix, (4, 6, 25)), atol=1e-4)
    np.testing.assert_allclose(norm.normalize(prefix), cands.prefix, atol=1e-5)
    np.testing.assert_allclose(norm.unnormalize(cands.normalized), cands.robot, atol=1e-4)
    # Free rows differ across samples; unconditioned sampling works too.
    assert np.abs(cands.normalized[0, 6:] - cands.normalized[1, 6:]).max() > 1e-3
    assert sampler.sample(images, state, "stack the cubes", None).prefix_len == 0


def test_latency_n8(served):
    from expo_ft.ultra.sampler import RtcCandidateSampler

    policy, _, images, state, prefix, _ = served
    s8 = RtcCandidateSampler(policy._policy, robot_action_dim=25, num_steps=10, n=8)
    s8.sample(images, state, "stack the cubes", prefix)  # compile
    t = []
    for _ in range(10):
        t0 = time.perf_counter()
        s8.sample(images, state, "stack the cubes", prefix)
        t.append(time.perf_counter() - t0)
    print(f"\nN=8 RTC candidate sampling: median {1e3 * np.median(t):.1f} ms, max {1e3 * max(t):.1f} ms")
