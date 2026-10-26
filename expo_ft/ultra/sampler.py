"""N base chunks per observation from a train-time-RTC pi0.5, sharing one prefix pass.

``Pi0.sample_actions`` (lair-openpi models/pi0.py) runs the expensive conditioning pass
(SigLIP + Gemma over images and prompt) once per batch element. All N candidates
condition on the same observation, so here that pass runs once and its KV cache is tiled
N times; only the flow integration (the action expert) runs at batch N. Prefix pinning is
Pi0.sample_actions' own: the first d rows are held at the committed actions with time 0.

The openpi ``Policy`` (the one server_openpi builds) supplies the input transforms
(normalization, tokenization, resize) and output transforms, so candidates are exactly what
``Policy.infer`` would return for the same noise -- ``tests/ultra/test_sampler.py`` checks
that on a real checkpoint.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np


@dataclasses.dataclass
class Candidates:
    normalized: np.ndarray  # (N, H, A) model space, robot dims only
    robot: np.ndarray  # (N, H, A) robot space
    state: np.ndarray  # (A,) normalized state (the VLA's own state input)
    prefix_len: int
    prefix: np.ndarray  # (d, A) normalized committed rows
    model_state: np.ndarray  # the transformed state, as the output transforms want it


class RtcCandidateSampler:
    def __init__(self, policy: Any, *, robot_action_dim: int, num_steps: int, n: int, seed: int = 0):
        import flax.nnx as nnx
        import jax

        model = policy._model  # noqa: SLF001
        assert getattr(model, "rtc_train_enabled", False), "the base policy must be served with --rtc-train"
        self._policy = policy
        self.robot_action_dim = int(robot_action_dim)
        self.num_steps = int(num_steps)
        self.n = int(n)
        self.horizon = int(model.action_horizon)
        self.model_action_dim = int(model.action_dim)
        self._rng = jax.random.PRNGKey(seed)
        graphdef, state = nnx.split(model)

        def sample_fn(state, observation, noise, action_prefix, prefix_len):
            return _sample_shared_prefix(nnx.merge(graphdef, state), observation, noise,
                                         action_prefix, prefix_len, self.num_steps)

        jitted = jax.jit(sample_fn)
        self._sample = lambda *a: jitted(state, *a)

    def transform(self, images: dict[str, np.ndarray], state_robot: np.ndarray, prompt: str,
                  prefix_robot: Optional[np.ndarray]) -> tuple[dict, int]:
        """Raw obs + robot-space prefix -> transformed (normalized) model inputs and d."""
        import jax

        padded = np.zeros((self.horizon, self.robot_action_dim), np.float32)
        d = 0
        if prefix_robot is not None and len(prefix_robot):
            d = min(len(prefix_robot), self.horizon)
            padded[:d] = prefix_robot[:d]
        obs = {
            "state": np.asarray(state_robot, np.float32),
            "image": images,
            "image_mask": {k: np.True_ for k in images},
            "prompt": prompt,
            "actions": padded,
        }
        return self._policy._input_transform(jax.tree.map(lambda x: x, obs)), d  # noqa: SLF001

    def sample(self, images: dict[str, np.ndarray], state_robot: np.ndarray, prompt: str,
               prefix_robot: Optional[np.ndarray] = None, noise: Optional[np.ndarray] = None) -> Candidates:
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        inputs, d = self.transform(images, state_robot, prompt, prefix_robot)
        action_prefix = jnp.asarray(inputs.pop("actions"))[None]  # (1, H, model dim) normalized
        batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(batched)
        if noise is None:
            self._rng, key = jax.random.split(self._rng)
            noise = jax.random.normal(key, (self.n, self.horizon, self.model_action_dim))
        x0 = self._sample(observation, jnp.asarray(noise, jnp.float32), action_prefix, jnp.asarray(d, jnp.int32))
        x0 = np.asarray(x0, np.float32)  # (N, H, model dim)
        robot = self.unnormalize(x0, inputs["state"])
        a = self.robot_action_dim
        return Candidates(normalized=x0[..., :a], robot=robot, state=np.asarray(inputs["state"])[:a],
                          prefix_len=d, prefix=np.asarray(action_prefix[0, :d, :a]),
                          model_state=np.asarray(inputs["state"]))

    def unnormalize(self, chunks_norm: np.ndarray, state_norm: np.ndarray) -> np.ndarray:
        """(N, H, <=model dim) normalized -> (N, H, A) robot space via the policy's output transforms."""
        out = []
        for c in chunks_norm:
            padded = np.zeros((c.shape[0], self.model_action_dim), np.float32)
            padded[:, : c.shape[1]] = c
            res = self._policy._output_transform({"state": np.asarray(state_norm), "actions": padded})  # noqa: SLF001
            out.append(np.asarray(res["actions"])[:, : self.robot_action_dim])
        return np.stack(out).astype(np.float32)


def _sample_shared_prefix(model, observation, noise, action_prefix, prefix_len, num_steps: int):
    """Pi0.sample_actions with the prefix (conditioning) pass at batch 1 and the flow at batch N."""
    import einops
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    n = noise.shape[0]
    observation = _model.preprocess_observation(None, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    # Tile everything the suffix reads to batch N. KV cache layout is (layers, batch, ...).
    kv_cache = jax.tree.map(lambda x: jnp.repeat(x, n, axis=1), kv_cache)
    prefix_mask = jnp.repeat(prefix_mask, n, axis=0)
    observation = jax.tree.map(lambda x: jnp.repeat(x, n, axis=0), observation)
    action_prefix = jnp.repeat(action_prefix, n, axis=0).astype(noise.dtype)
    pinned = jnp.arange(model.action_horizon)[None, :] < prefix_len  # (1, H)
    pinned = jnp.broadcast_to(pinned, (n, model.action_horizon))
    dt = -1.0 / num_steps

    def with_prefix(x_t, time):
        return jnp.where(pinned[..., None], action_prefix, x_t), jnp.where(pinned, 0.0, time)

    def step(carry):
        x_t, time = carry
        x_t, timestep = with_prefix(x_t, time)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, timestep)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        p_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([p_mask, suffix_attn_mask], axis=-1)
        pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens], mask=full_attn_mask, positions=pos, kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond])
        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon:])
        return x_t + dt * v_t, time + dt

    def cond(carry):
        return carry[1] >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return with_prefix(x_0, 0.0)[0]
