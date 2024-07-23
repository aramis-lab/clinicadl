from torch import Tensor
from torch.nn import MultiMarginLoss

from clinicadl.losses import ImplementedLoss, LossConfig, get_loss_function


def test_get_loss_function():
    for loss in [e.value for e in ImplementedLoss]:
        config = LossConfig(loss=loss)
        get_loss_function(config)

    config = LossConfig(loss="MultiMarginLoss", reduction="sum", weight=[1, 2, 3], p=2)
    loss, config_dict = get_loss_function(config)
    assert isinstance(loss, MultiMarginLoss)
    assert loss.reduction == "sum"
    assert loss.p == 2
    assert loss.margin == 1.0
    assert (loss.weight == Tensor([1, 2, 3])).all()
    assert config_dict == {
        "loss": "MultiMarginLoss",
        "reduction": "sum",
        "p": 2,
        "margin": 1.0,
        "weight": [1, 2, 3],
        "size_average": None,
        "reduce": None,
    }
