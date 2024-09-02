import pytest
from pydantic import ValidationError

from clinicadl.optim.lr_scheduler import LRSchedulerConfig


def test_LRSchedulerConfig():
    config = LRSchedulerConfig(
        scheduler="ReduceLROnPlateau",
        mode="max",
        patience=1,
        threshold_mode="rel",
        milestones=[4, 3, 2],
        min_lr={"param_0": 1e-1, "ELSE": 1e-2},
    )
    assert config.scheduler == "ReduceLROnPlateau"
    assert config.mode == "max"
    assert config.patience == 1
    assert config.threshold_mode == "rel"
    assert config.milestones == [2, 3, 4]
    assert config.min_lr == {"param_0": 1e-1, "ELSE": 1e-2}
    assert config.threshold == "DefaultFromLibrary"

    with pytest.raises(ValidationError):
        LRSchedulerConfig(last_epoch=-2)
    with pytest.raises(ValueError):
        LRSchedulerConfig(scheduler="abc")
    with pytest.raises(ValueError):
        LRSchedulerConfig(mode="abc")
    with pytest.raises(ValueError):
        LRSchedulerConfig(threshold_mode="abc")
    with pytest.raises(ValidationError):
        LRSchedulerConfig(milestones=[10, 10])
    with pytest.raises(ValidationError):
        LRSchedulerConfig(scheduler="MultiStepLR")
    with pytest.raises(ValidationError):
        LRSchedulerConfig(scheduler="StepLR")
