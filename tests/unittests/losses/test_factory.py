import pytest


def test_get_loss_function():
    from torch.nn import MultiMarginLoss

    from clinicadl.losses import ImplementedLoss, LossConfig, get_loss_function

    for loss in [e.value for e in ImplementedLoss]:
        config = LossConfig(loss=loss)
        get_loss_function(config)

    config = LossConfig(loss="MultiMarginLoss", reduction="sum", weight=[1, 2, 3])
    loss, config_dict = get_loss_function(config)
    assert isinstance(loss, MultiMarginLoss)
    assert config_dict == {
        "loss": "MultiMarginLoss",
        "reduction": "sum",
        "p": 1,
        "margin": 1.0,
        "weight": [1, 2, 3],
    }
