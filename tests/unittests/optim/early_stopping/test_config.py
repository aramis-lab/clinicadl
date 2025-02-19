import pytest
from pydantic import ValidationError

from clinicadl.optim.early_stopping import EarlyStopping


def test_EarlyStoppingConfig():
    earlystopping = EarlyStopping(
        patience=10,
        mode="max",
        check_finite=False,
        upper_bound=10.0,
    )
    assert earlystopping.patience == 10
    assert earlystopping.mode == "max"
    assert not earlystopping.check_finite
    assert earlystopping.upper_bound == 10.0

    with pytest.raises(ValueError):
        EarlyStopping(mode="abc")
    with pytest.raises(ValidationError):
        EarlyStopping(upper_bound=0.9, lower_bound=1.0)
