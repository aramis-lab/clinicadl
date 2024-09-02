import pytest
from pydantic import ValidationError

from clinicadl.optim.early_stopping import EarlyStoppingConfig


def test_EarlyStoppingConfig():
    config = EarlyStoppingConfig(
        patience=10,
        mode="max",
        check_finite=False,
        upper_bound=10.0,
    )
    assert config.patience == 10
    assert config.mode == "max"
    assert not config.check_finite
    assert config.upper_bound == 10.0

    with pytest.raises(ValueError):
        EarlyStoppingConfig(mode="abc")
    with pytest.raises(ValidationError):
        EarlyStoppingConfig(upper_bound=0.9, lower_bound=1.0)
