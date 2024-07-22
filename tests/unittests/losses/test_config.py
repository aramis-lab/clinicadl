import pytest


def test_LossConfig():
    from clinicadl.losses import LossConfig

    LossConfig(reduction="none", p=2, weight=[0.1, 0.1, 0.8])
    with pytest.raises(ValueError):
        LossConfig(loss="abc")
    with pytest.raises(ValueError):
        LossConfig(weight=[0.1, -0.1, 0.8])
