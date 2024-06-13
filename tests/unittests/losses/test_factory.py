import pytest


def test_get_loss_function():
    from clinicadl.losses import ImplementedLoss, get_loss_function

    for loss in [e.value for e in ImplementedLoss]:
        get_loss_function(loss)
    with pytest.raises(ValueError):
        get_loss_function("abc")
