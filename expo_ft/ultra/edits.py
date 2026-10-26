"""Knot-parameterized chunk edits.

The edit policy outputs u in [-1, 1]^(K x E): one offset per knot (K knots spread over the
C executed rows) per edited dimension (E = the dims with non-zero scale). An offset chunk
for H rows is built from u by

    offset[r, dims] = envelope(r, d) * scale[dims] * (W @ u)[r]

where W (H x K) linearly interpolates between knots placed at rows linspace(0, C-1, K) and
holds the last knot's value for every row >= C-1. Holding it means the rows past the
executed window -- the next decision's RTC prefix -- carry the same offset the robot was
executing, so the next chunk starts where this one left off instead of snapping back to
the base plan. The envelope zeroes the d committed prefix rows (they are already being
executed and must stay pinned) and fades the edit in over `ramp` rows after them.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from expo_ft.ultra.config import ExpoFtConfig


def knot_rows(exec_steps: int, num_knots: int) -> np.ndarray:
    if num_knots == 1:
        return np.zeros(1, np.float32)
    return np.linspace(0.0, exec_steps - 1, num_knots, dtype=np.float32)


def interpolation_matrix(exec_steps: int, num_knots: int, rows: int) -> np.ndarray:
    """(rows, num_knots) linear-interpolation weights; rows >= exec_steps-1 copy the last knot."""
    knots = knot_rows(exec_steps, num_knots)
    r = np.minimum(np.arange(rows, dtype=np.float32), exec_steps - 1)
    w = np.zeros((rows, num_knots), np.float32)
    if num_knots == 1:
        w[:, 0] = 1.0
        return w
    seg = np.clip(np.searchsorted(knots, r, side="right") - 1, 0, num_knots - 2)
    frac = (r - knots[seg]) / (knots[seg + 1] - knots[seg])
    w[np.arange(rows), seg] = 1.0 - frac
    w[np.arange(rows), seg + 1] = frac
    return w


def row_envelope(prefix_len, rows: int, ramp: int):
    """(..., rows): 0 on the committed prefix rows [0, d), then a linear fade-in to 1."""
    r = jnp.arange(rows, dtype=jnp.float32)
    d = jnp.asarray(prefix_len, jnp.float32)[..., None]
    if ramp <= 0:
        return (r >= d).astype(jnp.float32)
    return jnp.clip((r - d + 1.0) / ramp, 0.0, 1.0)


class EditExpander:
    """Turns raw edit samples u (..., K*E) into offset chunks (..., rows, action_dim)."""

    def __init__(self, cfg: ExpoFtConfig):
        self.cfg = cfg
        self.dims = np.asarray(cfg.edit_dims, np.int32)
        self.scale = cfg.edit_scale_vector[self.dims]  # (E,)
        self.num_knots = cfg.num_knots

    def log_scale_sum(self) -> float:
        """log|det| of u -> scale * u for the change-of-variables term in log pi (all knots)."""
        return float(self.num_knots * np.sum(np.log(self.scale)))

    def __call__(self, u, prefix_len, rows: int):
        cfg = self.cfg
        k, e = self.num_knots, len(self.dims)
        u = u.reshape(*u.shape[:-1], k, e)
        w = jnp.asarray(interpolation_matrix(cfg.exec_steps, k, rows))
        per_row = jnp.einsum("rk,...ke->...re", w, u, precision="highest") * self.scale  # (..., rows, E)
        per_row = per_row * row_envelope(prefix_len, rows, cfg.edit_ramp_rows)[..., None]
        out = jnp.zeros((*per_row.shape[:-1], cfg.action_dim), per_row.dtype)
        return out.at[..., self.dims].set(per_row)
