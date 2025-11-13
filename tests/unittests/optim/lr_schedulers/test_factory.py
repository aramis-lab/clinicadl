import pytest

from clinicadl.optim.lr_schedulers.config import *
from clinicadl.optim.lr_schedulers.factory import get_lr_scheduler_from_dict

MANDATORY_FIELDS = {
    "step_size": 1,
    "milestones": [1, 2],
    "max_lr": 1,
    "total_steps": 10,
    "gamma": 1,
}


@pytest.mark.parametrize(
    "config",
    [
        ConstantLRConfig,
        ExponentialLRConfig,
        LinearLRConfig,
        MultiStepLRConfig,
        OneCycleLRConfig,
        PolynomialLRConfig,
        ReduceLROnPlateauConfig,
        StepLRConfig,
    ],
)
def test_get_transform_config(config):
    c = config(**MANDATORY_FIELDS)
    config_dict = c.to_dict()
    c = get_lr_scheduler_from_dict(config_dict)
    assert isinstance(c, config)

    if config is StepLRConfig:
        c = StepLRConfig(step_size=1)
        assert get_lr_scheduler_from_dict(c.to_dict()).step_size == 1
