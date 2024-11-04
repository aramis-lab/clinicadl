from torch import Tensor
from torch.nn import BCEWithLogitsLoss, MultiMarginLoss

from clinicadl.losses import (
    ImplementedLoss,
    get_loss_function,
    get_loss_function_from_config,
)
from clinicadl.losses.config import MultiMarginLossConfig


def test_get_loss_function():
    for loss in ImplementedLoss:
        _ = get_loss_function(loss)


def test_parameters():
    loss, config = get_loss_function(
        name="MultiMarginLoss",
        return_config=True,
        weight=[1, 2, 3],
        p=2,
        margin=1.0,
        reduction="sum",
    )
    assert isinstance(loss, MultiMarginLoss)
    assert loss.reduction == "sum"
    assert loss.p == 2
    assert loss.margin == 1.0
    assert (loss.weight == Tensor([1, 2, 3])).all()

    assert config.name == "MultiMarginLoss"
    assert config.reduction == "sum"
    assert config.p == 2
    assert config.margin == 1.0
    assert config.weight == [1, 2, 3]

    loss, config = get_loss_function(
        "BCEWithLogitsLoss", return_config=True, pos_weight=[1, 2, 3]
    )
    assert isinstance(loss, BCEWithLogitsLoss)
    assert (loss.pos_weight == Tensor([[1, 2, 3]])).all()
    assert config.pos_weight == [1, 2, 3]


def test_without_return():
    net = get_loss_function(
        "MultiMarginLoss",
    )
    assert isinstance(net, MultiMarginLoss)


def test_get_loss_function_from_config():
    config = MultiMarginLossConfig()
    loss, updated_config = get_loss_function_from_config(config)
    assert isinstance(loss, MultiMarginLoss)
    assert updated_config.margin == 1.0
    assert config.margin == "DefaultFromLibrary"
