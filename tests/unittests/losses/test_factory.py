import pytest
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, MultiMarginLoss

from clinicadl.losses import (
    get_loss_function_config,
    get_loss_function_from_config,
)
from clinicadl.losses.config import (
    ImplementedLoss,
    create_loss_function_config,
)


def test_get_loss_function_from_config():
    # test all optimizers
    for loss_name in ImplementedLoss:
        config = create_loss_function_config(loss_name)()
        loss, _ = get_loss_function_from_config(config=config)

    # test arguments
    config = create_loss_function_config("MultiMarginLoss")(
        weight=[1, 2, 3],
        margin=1.0,
        reduction="sum",
    )
    loss, updated_config = get_loss_function_from_config(config)
    assert isinstance(loss, MultiMarginLoss)
    assert loss.reduction == "sum"
    assert loss.margin == 1.0
    assert (loss.weight == Tensor([1, 2, 3])).all()

    assert updated_config.name == "MultiMarginLoss"
    assert updated_config.reduction == "sum"
    assert updated_config.p == 1
    assert updated_config.margin == 1.0
    assert updated_config.weight == [1, 2, 3]

    # special case
    config = create_loss_function_config("BCEWithLogitsLoss")(pos_weight=[1, 2, 3])
    loss, updated_config = get_loss_function_from_config(config)
    assert isinstance(loss, BCEWithLogitsLoss)
    assert (loss.pos_weight == Tensor([[1, 2, 3]])).all()
    assert config.pos_weight == [1, 2, 3]


def test_get_loss_function_config():
    config = get_loss_function_config(
        "MultiMarginLoss", weight=[1, 2, 3], margin=1.0, reduction="sum"
    )
    assert config.name == "MultiMarginLoss"
    assert config.reduction == "sum"
    assert config.p == 1
    assert config.margin == 1.0
    assert config.weight == [1, 2, 3]

    with pytest.raises(ValueError):
        get_loss_function_config("abc", weight=[1, 2, 3], margin=1.0, reduction="sum")
