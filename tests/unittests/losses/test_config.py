import pytest
from pydantic import ValidationError

from clinicadl.losses import LossConfig


def test_LossConfig():
    config = LossConfig(
        loss="SmoothL1Loss", margin=10.0, delta=2.0, reduction="none", weight=None
    )
    assert config.loss == "SmoothL1Loss"
    assert config.margin == 10.0
    assert config.delta == 2.0
    assert config.reduction == "none"
    assert config.p == "DefaultFromLibrary"

    with pytest.raises(ValueError):
        LossConfig(loss="abc")
    with pytest.raises(ValueError):
        LossConfig(weight=[0.1, -0.1, 0.8])
    with pytest.raises(ValueError):
        LossConfig(p=3)
    with pytest.raises(ValueError):
        LossConfig(reduction="abc")
    with pytest.raises(ValidationError):
        LossConfig(label_smoothing=1.1)
    with pytest.raises(ValidationError):
        LossConfig(ignore_index=-1)
    with pytest.raises(ValidationError):
        LossConfig(loss="BCEWithLogitsLoss", weight=[1, 2, 3])
    with pytest.raises(ValidationError):
        LossConfig(loss="BCELoss", weight=[1, 2, 3])

    LossConfig(loss="BCELoss")
    LossConfig(loss="BCEWithLogitsLoss", weight=None)
