import dataclasses

import pytest

from expo_ft.ultra.config import ExpoFtConfig


def tiny_config(**overrides) -> ExpoFtConfig:
    """Real chunk geometry (A=25, H=50, C=20) with toy networks so tests run in seconds."""
    base = dict(
        image_size=32,
        encoder_stage_sizes=(1,),
        encoder_num_filters=4,
        image_latent_dim=16,
        state_latent_dim=8,
        hidden_dims=(32, 32),
        num_qs=4,
        batch_size=4,
        utd_ratio=2,
        augment_pad=2,
    )
    base.update(overrides)
    return ExpoFtConfig(**base)


@pytest.fixture
def cfg():
    return tiny_config()


def replace(cfg, **kw):
    return dataclasses.replace(cfg, **kw)
