import pytest
import torch
from pydantic import ValidationError

from clinicadl.losses import ImplementedLoss
from clinicadl.losses.config import (
    BCEConfig,
    BCEWithLogitsConfig,
    CrossEntropyConfig,
    HuberConfig,
    KLDivConfig,
    L1Config,
    MSEConfig,
    MultiMarginConfig,
    NLLConfig,
    SmoothL1Config,
    create_loss_config,
)


@pytest.mark.parametrize(
    "config,args",
    [
        (L1Config, {"reduction": "none"}),
        (MSEConfig, {"reduction": "none"}),
        (CrossEntropyConfig, {"reduction": "none"}),
        (CrossEntropyConfig, {"weight": [1, -1, 2]}),
        (CrossEntropyConfig, {"ignore_index": -1}),
        (CrossEntropyConfig, {"label_smoothing": 1.1}),
        (NLLConfig, {"reduction": "none"}),
        (NLLConfig, {"weight": [1, -1, 2]}),
        (NLLConfig, {"ignore_index": -1}),
        (KLDivConfig, {"reduction": "none"}),
        (BCEConfig, {"reduction": "none"}),
        (BCEConfig, {"weight": [0, 1]}),
        (BCEWithLogitsConfig, {"reduction": "none"}),
        (BCEWithLogitsConfig, {"weight": [0, 1]}),
        (BCEWithLogitsConfig, {"pos_weight": [[1, -1, 2]]}),
        (BCEWithLogitsConfig, {"pos_weight": [["a", "b"]]}),
        (HuberConfig, {"reduction": "none"}),
        (HuberConfig, {"delta": 0.0}),
        (SmoothL1Config, {"reduction": "none"}),
        (SmoothL1Config, {"beta": -1.0}),
        (MultiMarginConfig, {"reduction": "none"}),
        (MultiMarginConfig, {"p": 3}),
        (MultiMarginConfig, {"weight": [1, -1, 2]}),
    ],
)
def test_validation_fail(config, args):
    with pytest.raises(ValidationError):
        config(**args)


@pytest.mark.parametrize(
    "config,args",
    [
        (L1Config, {"reduction": "mean"}),
        (MSEConfig, {"reduction": "mean"}),
        (
            CrossEntropyConfig,
            {
                "reduction": "mean",
                "weight": [1, 0, 2],
                "ignore_index": 1,
                "label_smoothing": 0.5,
            },
        ),
        (NLLConfig, {"reduction": "mean", "weight": [1, 0, 2], "ignore_index": 1}),
        (KLDivConfig, {"reduction": "mean", "log_target": True}),
        (BCEConfig, {"reduction": "sum", "weight": None}),
        (
            BCEWithLogitsConfig,
            {"reduction": "sum", "weight": None, "pos_weight": [[1, 0, 2]]},
        ),
        (HuberConfig, {"reduction": "sum", "delta": 0.1}),
        (SmoothL1Config, {"reduction": "sum", "beta": 0.0}),
        (
            MultiMarginConfig,
            {"reduction": "sum", "p": 1, "margin": -0.1, "weight": [1, 0, 2]},
        ),
    ],
)
def test_validation_pass(config, args):
    c = config(**args)
    for arg, value in args.items():
        assert getattr(c, arg) == value


def test_create_loss_config():
    for loss in ImplementedLoss:
        create_loss_config(loss)

    config_class = create_loss_config("Multi Margin")
    config = config_class(
        margin=0.1,
        reduction="sum",
    )
    assert isinstance(config, MultiMarginConfig)
    assert config.p == "DefaultFromLibrary"
    assert config.margin == 0.1
    assert config.reduction == "sum"
