"""EXPO-FT learner for chunk-level decisions (frozen VLA, phase 1).

Pure JAX; no openpi imports. The VLA's base samples enter as arrays: at rollout time from
the sampler, and in the TD target from the per-transition candidate cache.

    select()  On-the-fly policy: N base chunks + N edits of them, argmax of a REDQ
              min-of-2 target Q. The same rule picks a* in the TD target.
    update()  `utd_ratio` critic steps (Eq. 5, chunk TD with gamma**n), then one edit
              actor step (Eq. 4) and one temperature step on the last minibatch -- the
              order of the authors' EXPOLearner._update_jit.

Deviation from the authors' code, on purpose: the critic image encoder lives inside the
critic's parameters, so the target network has a Polyak-averaged encoder too (theirs
re-encodes next observations with the online encoder). The TD target and rollout
selection run entirely on the target network, including the edit policy's input
features, which saves one ResNet pass over the next observations per step.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState

from expo_ft.networks.temperature import Temperature
from expo_ft.ultra.config import ExpoFtConfig
from expo_ft.ultra.edits import EditExpander
from expo_ft.ultra.networks import Critic, build_critic, build_edit_policy, tanh_gaussian_sample


class ExpoState(struct.PyTreeNode):
    critic: TrainState
    target_critic_params: Any
    edit: TrainState
    temp: TrainState
    rng: jax.Array
    step: jax.Array


def random_shift(key, images, pad: int):
    """DrQ random shift, independently per image. images (..., H, W, 3) uint8."""
    if pad <= 0:
        return images
    lead, (h, w, c) = images.shape[:-3], images.shape[-3:]
    flat = images.reshape((-1, h, w, c))
    keys = jax.random.split(key, flat.shape[0])

    def one(k, img):
        padded = jnp.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        off = jax.random.randint(k, (2,), 0, 2 * pad + 1)
        return jax.lax.dynamic_slice(padded, (off[0], off[1], 0), (h, w, c))

    return jax.vmap(one)(keys, flat).reshape(images.shape)


def _subset_min(key, qs, num_qs: int, num_min_qs: int):
    """REDQ: min over a random subset of the ensemble. qs (num_qs, ...) -> (...)."""
    idx = jax.random.choice(key, num_qs, (num_min_qs,), replace=False)
    return qs[idx].min(axis=0)


class ExpoLearner:
    """Holds the static pieces (config, module definitions) and the jitted functions."""

    def __init__(self, cfg: ExpoFtConfig, encoder=None):
        self.cfg = cfg
        self.critic_def: Critic = build_critic(cfg, encoder)
        self.edit_def = build_edit_policy(cfg)
        self.temp_def = Temperature(cfg.init_temperature)
        self.expander = EditExpander(cfg)
        self.log_scale_sum = self.expander.log_scale_sum()
        self.target_entropy = cfg.resolved_target_entropy
        self.select = jax.jit(self._select)
        self.update = jax.jit(self._update)
        self.td_target = jax.jit(self._td_target)

    # ------------------------------------------------------------------ init
    def init(self, rng) -> ExpoState:
        cfg = self.cfg
        rng, k_c, k_e, k_t = jax.random.split(rng, 4)
        v, s = len(cfg.camera_keys), cfg.image_size
        images = jnp.zeros((1, v, s, s, 3), jnp.uint8)
        state = jnp.zeros((1, cfg.action_dim))
        actions = jnp.zeros((1, cfg.chunk_flat_dim))
        critic_params = self.critic_def.init(k_c, images, state, actions)["params"]
        feats = self.critic_def.apply({"params": critic_params}, images, state, method=Critic.features)
        edit_params = self.edit_def.init(k_e, feats, actions)["params"]
        temp_params = self.temp_def.init(k_t)["params"]
        tx = lambda: optax.adam(cfg.lr)  # noqa: E731
        state = ExpoState(
            critic=TrainState.create(apply_fn=self.critic_def.apply, params=critic_params, tx=tx()),
            target_critic_params=critic_params,
            edit=TrainState.create(apply_fn=self.edit_def.apply, params=edit_params, tx=tx()),
            temp=TrainState.create(apply_fn=self.temp_def.apply, params=temp_params, tx=tx()),
            rng=rng,
            step=jnp.zeros((), jnp.int32),
        )
        # Strongly typed device arrays everywhere (TrainState.step starts as a Python int and
        # the temperature as a weak float): otherwise the state that comes back from the
        # first update has a different signature and select/update compile a second time.
        return jax.tree.map(lambda x: jnp.asarray(np.asarray(x)), state)

    # ------------------------------------------------------------------ pieces
    def _features(self, params, images, state):
        return self.critic_def.apply({"params": params}, images, state, method=Critic.features)

    def _qs(self, params, feats, actions_flat):
        return self.critic_def.apply({"params": params}, feats, actions_flat, method=Critic.q_from_features)

    def _sample_edit(self, key, edit_params, feats, base, prefix_len, rows: int):
        """feats (B, F), base (B, rows>=C, A) -> (edited (B, rows, A), u, log_prob)."""
        c = self.cfg.exec_steps
        mean, log_std = self.edit_def.apply({"params": edit_params}, feats, base[:, :c].reshape(base.shape[0], -1))
        u, log_prob = tanh_gaussian_sample(key, mean, log_std)
        offsets = self.expander(u, prefix_len, rows)
        return base + offsets, u, log_prob - self.log_scale_sum

    def _otf_choose(self, key, critic_params, edit_params, feats, base, prefix_len):
        """The on-the-fly policy, batched.

        feats (B, F); base (B, N, R, A) base chunks with R >= C rows; prefix_len (B,).
        Returns (best (B, R, A), idx (B,), qs (B, N + n_edit), candidates (B, N+n_edit, R, A)).
        """
        cfg = self.cfg
        b, n, rows, a = base.shape
        k_edit, k_sel = jax.random.split(key)
        n_edit = cfg.n_edit
        if n_edit > 0:
            src = base[:, :n_edit].reshape(b * n_edit, rows, a)
            f_rep = jnp.repeat(feats, n_edit, axis=0)
            d_rep = jnp.repeat(prefix_len, n_edit, axis=0)
            edited, _, _ = self._sample_edit(k_edit, edit_params, f_rep, src, d_rep, rows)
            cands = jnp.concatenate([base, edited.reshape(b, n_edit, rows, a)], axis=1)
        else:
            cands = base
        m = cands.shape[1]
        flat = cands[:, :, : cfg.exec_steps].reshape(b * m, cfg.chunk_flat_dim)
        qs = self._qs(critic_params, jnp.repeat(feats, m, axis=0), flat)  # (num_qs, B*m)
        q = _subset_min(k_sel, qs, cfg.num_qs, cfg.num_min_qs).reshape(b, m)
        idx = jnp.argmax(q, axis=1)
        best = jnp.take_along_axis(cands, idx[:, None, None, None], axis=1)[:, 0]
        return best, idx, q, cands

    # ------------------------------------------------------------------ rollout
    def _select(self, state: ExpoState, images, robot_state, base, prefix_len, key):
        """One decision. images (V, h, w, 3) uint8, robot_state (A,) normalized,
        base (N, H, A) normalized base chunks (prefix already pinned), prefix_len scalar.
        Returns dict(chunk (H, A), idx, qs (N+n_edit,), candidates (N+n_edit, H, A))."""
        feats = self._features(state.target_critic_params, images[None], robot_state[None])
        best, idx, q, cands = self._otf_choose(
            key, state.target_critic_params, state.edit.params, feats, base[None], jnp.asarray(prefix_len)[None])
        return {"chunk": best[0], "idx": idx[0], "qs": q[0], "candidates": cands[0]}

    # ------------------------------------------------------------------ training
    def _td_target(self, state: ExpoState, batch, key):
        """Eq. 5: r + gamma**n * mask * min-of-2 Q'(s', a*), a* = OTF argmax at s'."""
        cfg = self.cfg
        k_choose, k_eval = jax.random.split(key)
        feats = self._features(state.target_critic_params, batch["next_images"], batch["next_state"])
        best, idx, _, _ = self._otf_choose(
            k_choose, state.target_critic_params, state.edit.params, feats,
            batch["next_base"], batch["next_prefix_len"])
        next_qs = self._qs(state.target_critic_params, feats, best.reshape(best.shape[0], -1))
        next_q = _subset_min(k_eval, next_qs, cfg.num_qs, cfg.num_min_qs)
        target = batch["rewards"] + batch["discounts"] * next_q
        info = {"target_q": target.mean(), "next_q": next_q.mean(),
                "select_edit_ratio": jnp.mean((idx >= cfg.n_base).astype(jnp.float32))}
        return jax.lax.stop_gradient(target), info

    def _critic_step(self, state: ExpoState, batch):
        cfg = self.cfg
        rng, k_aug, k_aug2, k_td = jax.random.split(state.rng, 4)
        batch = dict(batch)
        batch["images"] = random_shift(k_aug, batch["images"], cfg.augment_pad)
        batch["next_images"] = random_shift(k_aug2, batch["next_images"], cfg.augment_pad)
        target, td_info = self._td_target(state, batch, k_td)

        def loss_fn(params):
            feats = self._features(params, batch["images"], batch["state"])
            qs = self._qs(params, feats, batch["actions"])  # (num_qs, B)
            loss = jnp.mean((qs - target[None]) ** 2)
            return loss, {"critic_loss": loss, "q": qs.mean(), "q_std_ensemble": qs.std(axis=0).mean()}

        grads, info = jax.grad(loss_fn, has_aux=True)(state.critic.params)
        critic = state.critic.apply_gradients(grads=grads)
        target_params = optax.incremental_update(critic.params, state.target_critic_params, cfg.tau_critic)
        info = {**info, **td_info, "critic_grad_norm": optax.global_norm(grads)}
        return state.replace(critic=critic, target_critic_params=target_params, rng=rng), info, batch

    def _edit_step(self, state: ExpoState, batch):
        """Eq. 4: max_pi E[Q(s, a + a_hat) - alpha log pi(a_hat | s, a)], a ~ base."""
        rng, key = jax.random.split(state.rng)
        feats = jax.lax.stop_gradient(self._features(state.critic.params, batch["images"], batch["state"]))
        alpha = self.temp_def.apply({"params": state.temp.params})

        def loss_fn(edit_params):
            edited, _, log_prob = self._sample_edit(
                key, edit_params, feats, batch["base"], batch["prefix_len"], self.cfg.exec_steps)
            qs = self._qs(state.critic.params, feats, edited.reshape(edited.shape[0], -1))
            q = qs.mean(axis=0)
            loss = jnp.mean(alpha * log_prob - q)
            return loss, {"edit_loss": loss, "edit_q": q.mean(), "entropy": -log_prob.mean()}

        grads, info = jax.grad(loss_fn, has_aux=True)(state.edit.params)
        edit = state.edit.apply_gradients(grads=grads)

        def temp_loss_fn(temp_params):
            a = self.temp_def.apply({"params": temp_params})
            return a * (info["entropy"] - self.target_entropy)

        t_grads = jax.grad(temp_loss_fn)(state.temp.params)
        temp = state.temp.apply_gradients(grads=t_grads)
        info = {**info, "temperature": alpha}
        return state.replace(edit=edit, temp=temp, rng=rng), info

    def _update(self, state: ExpoState, batch):
        """batch: arrays with a leading (utd_ratio * batch_size) axis (see buffer.sample)."""
        cfg = self.cfg
        u, b = cfg.utd_ratio, cfg.batch_size
        minis = jax.tree.map(lambda x: x.reshape((u, b) + x.shape[1:]), batch)

        def body(st, mb):
            st, info, _ = self._critic_step(st, mb)
            return st, info

        state, infos = jax.lax.scan(body, state, minis)
        last = jax.tree.map(lambda x: x[-1], minis)
        state, edit_info = self._edit_step(state, last)
        info = {**jax.tree.map(lambda x: x[-1], infos), **edit_info}
        return state.replace(step=state.step + 1), info


def param_count(tree) -> int:
    return int(sum(np.prod(x.shape) for x in jax.tree.leaves(tree)))


def to_host(info: dict) -> dict:
    return {k: float(v) for k, v in jax.device_get(info).items()}


