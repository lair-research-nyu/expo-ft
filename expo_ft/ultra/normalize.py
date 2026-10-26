"""Robot space <-> the VLA's normalized action space, in numpy.

Same formulas as openpi.transforms.Normalize/Unnormalize, so human actions, recorded
chunks and demo rows land exactly where the model's own outputs live. The sampler test
checks this against the checkpoint's real transform pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ActionNormalizer:
    def __init__(self, stats: dict, *, use_quantiles: bool = True, dim: int | None = None):
        self.use_quantiles = use_quantiles
        if use_quantiles:
            lo, hi = np.asarray(stats["q01"], np.float32), np.asarray(stats["q99"], np.float32)
            self._a = lo[:dim]
            self._b = (hi - lo)[:dim] + 1e-6
        else:
            self._a = np.asarray(stats["mean"], np.float32)[:dim]
            self._b = np.asarray(stats["std"], np.float32)[:dim] + 1e-6

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32)
        z = (x - self._a) / self._b
        return (z * 2.0 - 1.0) if self.use_quantiles else z

    def unnormalize(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, np.float32)
        if self.use_quantiles:
            z = (z + 1.0) / 2.0
        return z * self._b + self._a

    @classmethod
    def from_norm_stats_dir(cls, stats_dir: Path, key: str, *, use_quantiles: bool = True, dim: int | None = None):
        stats = json.loads((Path(stats_dir) / "norm_stats.json").read_text())["norm_stats"][key]
        return cls(stats, use_quantiles=use_quantiles, dim=dim)
