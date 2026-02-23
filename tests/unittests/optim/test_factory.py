import pytest

from clinicadl.optim.optimizers.config import *
from clinicadl.optim.optimizers.factory import get_optimizer_from_dict


@pytest.mark.parametrize(
    "config",
    [
        AdadeltaConfig,
        AdagradConfig,
        AdamConfig,
        RMSpropConfig,
        SGDConfig,
    ],
)
def test_get_optimizer_from_dict(config):
    c = config()
    config_dict = c.to_dict()
    c = get_optimizer_from_dict(config_dict)
    assert isinstance(c, config)

    if config is AdadeltaConfig:
        c = AdadeltaConfig(lr=0.5)
        assert get_optimizer_from_dict(c.to_dict()).lr == 0.5
