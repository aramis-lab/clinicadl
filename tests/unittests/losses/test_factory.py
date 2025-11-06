import pytest

from clinicadl.losses import get_loss_function_from_dict
from clinicadl.losses.config import *


@pytest.mark.parametrize(
    "config",
    [
        BCELossConfig,
        BCEWithLogitsLossConfig,
        CrossEntropyLossConfig,
        HuberLossConfig,
        KLDivLossConfig,
        L1LossConfig,
        MSELossConfig,
        MultiMarginLossConfig,
        NLLLossConfig,
        SmoothL1LossConfig,
    ],
)
def test_get_transform_config(config):
    c = config()
    config_dict = c.to_dict()
    c = get_loss_function_from_dict(config_dict)
    assert isinstance(c, config)

    if config is NLLLossConfig:
        c = NLLLossConfig(weight=[1, 2])
        assert get_loss_function_from_dict(c.to_dict()).weight == [1, 2]
