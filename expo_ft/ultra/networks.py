"""Critic, critic image encoder and edit policy.

Reuses the authors' Flax pieces (expo_ft.networks): ResNetV2Encoder, MLP, Ensemble,
StateActionValue. The tanh-Gaussian is written out here because expo_ft.distributions
depends on tensorflow_probability, which the lair-openpi venv does not have.

Shapes (B = batch, V = cameras, A = action dim, C = executed rows):
    images  (B, V, 224, 224, 3) uint8
    state   (B, A) normalized
    actions (B, C * A) normalized executed rows, flattened
"""

from __future__ import annotations

from functools import partial
from typing import Protocol, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from expo_ft.networks import MLP, Ensemble, StateActionValue, default_init
from expo_ft.networks.encoders import ResNetV2Encoder
from expo_ft.ultra.config import ExpoFtConfig

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


# ----------------------------------------------------------------------------------
# tanh-Gaussian
# ----------------------------------------------------------------------------------


def _softplus(x):
    return jnp.logaddexp(x, 0.0)


def tanh_gaussian_sample(key, mean, log_std):
    """Reparameterized sample u = tanh(mean + std * eps) and its log-density (summed)."""
    eps = jax.random.normal(key, mean.shape, mean.dtype)
    pre = mean + jnp.exp(log_std) * eps
    u = jnp.tanh(pre)
    log_prob = jnp.sum(-0.5 * eps**2 - log_std - 0.5 * jnp.log(2 * jnp.pi), axis=-1)
    # log|d tanh(x)/dx| = 2 * (log 2 - x - softplus(-2x)), numerically stable.
    log_prob -= jnp.sum(2.0 * (jnp.log(2.0) - pre - _softplus(-2.0 * pre)), axis=-1)
    return u, log_prob


# ----------------------------------------------------------------------------------
# critic image encoder (swappable)
# ----------------------------------------------------------------------------------


class CriticEncoder(Protocol):
    """Any Flax module mapping (B, V, H, W, 3) uint8 images to (B, F) features.

    The default is a from-scratch ResNet (Appendix D.1); a pretrained backbone (e.g. the
    value model's) plugs in by implementing the same call.
    """

    def __call__(self, images: jnp.ndarray) -> jnp.ndarray: ...


class SharedResNetEncoder(nn.Module):
    """One ResNetV2 trunk shared across cameras; each camera -> Dense -> LN -> tanh, concatenated."""

    stage_sizes: Sequence[int] = (3, 4, 6, 3)
    num_filters: int = 64
    latent_dim: int = 512

    @nn.compact
    def __call__(self, images: jnp.ndarray) -> jnp.ndarray:
        b, v = images.shape[:2]
        x = images.astype(jnp.float32) / 127.5 - 1.0
        x = x.reshape((b * v, *x.shape[2:]))
        x = ResNetV2Encoder(stage_sizes=tuple(self.stage_sizes), num_filters=self.num_filters)(x)
        x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        x = jnp.tanh(x)
        return x.reshape(b, v * self.latent_dim)


class StateEmbed(nn.Module):
    latent_dim: int = 64

    @nn.compact
    def __call__(self, state):
        y = nn.Dense(self.latent_dim, kernel_init=default_init())(state)
        return jnp.tanh(nn.LayerNorm()(y))


# ----------------------------------------------------------------------------------
# critic: Q(s, a_{t:t+C}) as a REDQ ensemble over shared image features
# ----------------------------------------------------------------------------------


class Critic(nn.Module):
    encoder: nn.Module
    hidden_dims: Sequence[int] = (256, 256, 256)
    num_qs: int = 10
    state_latent_dim: int = 64

    def setup(self):
        self.state_embed = StateEmbed(self.state_latent_dim)
        base = partial(MLP, hidden_dims=tuple(self.hidden_dims), activate_final=True, use_layer_norm=True)
        self.heads = Ensemble(partial(StateActionValue, base_cls=base), num=self.num_qs)

    def features(self, images, state):
        """Encoder output + state embedding: everything about s the heads see."""
        return jnp.concatenate([self.encoder(images), self.state_embed(state)], axis=-1)

    def q_from_features(self, feats, actions):
        """(num_qs, B) for features (B, F) and actions (B, C*A)."""
        return self.heads(feats, actions)

    def __call__(self, images, state, actions):
        return self.q_from_features(self.features(images, state), actions)


# ----------------------------------------------------------------------------------
# edit policy: pi_edit(u | s, a_base), u in [-1, 1]^(K*E)
# ----------------------------------------------------------------------------------


class EditPolicy(nn.Module):
    """Tanh-Gaussian over knot offsets, conditioned on critic features (stop-gradient,
    per the paper's B.2 ablation) and the base chunk's executed rows."""

    out_dim: int
    hidden_dims: Sequence[int] = (256, 256, 256)

    @nn.compact
    def __call__(self, feats, base_flat):
        x = jnp.concatenate([jax.lax.stop_gradient(feats), base_flat], axis=-1)
        x = MLP(hidden_dims=tuple(self.hidden_dims), activate_final=True)(x)
        mean = nn.Dense(self.out_dim, kernel_init=default_init(), name="mean")(x)
        log_std = nn.Dense(self.out_dim, kernel_init=default_init(), name="log_std")(x)
        return mean, jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)


def build_critic(cfg: ExpoFtConfig, encoder: nn.Module | None = None) -> Critic:
    if encoder is None:
        encoder = SharedResNetEncoder(
            stage_sizes=cfg.encoder_stage_sizes,
            num_filters=cfg.encoder_num_filters,
            latent_dim=cfg.image_latent_dim,
        )
    return Critic(encoder=encoder, hidden_dims=cfg.hidden_dims, num_qs=cfg.num_qs,
                  state_latent_dim=cfg.state_latent_dim)


def build_edit_policy(cfg: ExpoFtConfig) -> EditPolicy:
    return EditPolicy(out_dim=cfg.edit_param_dim, hidden_dims=cfg.hidden_dims)
