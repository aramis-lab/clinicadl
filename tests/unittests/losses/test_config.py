import pytest
from pydantic import ValidationError

from clinicadl.losses import ImplementedLoss
from clinicadl.losses.config import (
    BCELossConfig,
    BCEWithLogitsLossConfig,
    CrossEntropyLossConfig,
    HuberLossConfig,
    KLDivLossConfig,
    L1LossConfig,
    MSELossConfig,
    MultiMarginLossConfig,
    NLLLossConfig,
    SmoothL1LossConfig,
    create_loss_config,
)


@pytest.mark.parametrize(
    "config,args",
    [
        (L1LossConfig, {"reduction": "none"}),
        (MSELossConfig, {"reduction": "none"}),
        (CrossEntropyLossConfig, {"reduction": "none"}),
        (CrossEntropyLossConfig, {"weight": [1, -1, 2]}),
        (CrossEntropyLossConfig, {"ignore_index": -1}),
        (CrossEntropyLossConfig, {"label_smoothing": 1.1}),
        (NLLLossConfig, {"reduction": "none"}),
        (NLLLossConfig, {"weight": [1, -1, 2]}),
        (NLLLossConfig, {"ignore_index": -1}),
        (KLDivLossConfig, {"reduction": "none"}),
        (BCELossConfig, {"reduction": "none"}),
        (BCELossConfig, {"weight": [0, 1]}),
        (BCEWithLogitsLossConfig, {"reduction": "none"}),
        (BCEWithLogitsLossConfig, {"weight": [0, 1]}),
        (BCEWithLogitsLossConfig, {"pos_weight": [[1, -1, 2]]}),
        (BCEWithLogitsLossConfig, {"pos_weight": [["a", "b"]]}),
        (HuberLossConfig, {"reduction": "none"}),
        (HuberLossConfig, {"delta": 0.0}),
        (SmoothL1LossConfig, {"reduction": "none"}),
        (SmoothL1LossConfig, {"beta": -1.0}),
        (MultiMarginLossConfig, {"reduction": "none"}),
        (MultiMarginLossConfig, {"p": 3}),
        (MultiMarginLossConfig, {"weight": [1, -1, 2]}),
    ],
)
def test_validation_fail(config, args):
    with pytest.raises(ValidationError):
        config(**args)


@pytest.mark.parametrize(
    "config,args",
    [
        (L1LossConfig, {"reduction": "mean"}),
        (MSELossConfig, {"reduction": "mean"}),
        (
            CrossEntropyLossConfig,
            {
                "reduction": "mean",
                "weight": [1, 0, 2],
                "ignore_index": 1,
                "label_smoothing": 0.5,
            },
        ),
        (NLLLossConfig, {"reduction": "mean", "weight": [1, 0, 2], "ignore_index": 1}),
        (KLDivLossConfig, {"reduction": "mean", "log_target": True}),
        (BCELossConfig, {"reduction": "sum", "weight": None}),
        (
            BCEWithLogitsLossConfig,
            {"reduction": "sum", "weight": None, "pos_weight": [[1, 0, 2]]},
        ),
        (HuberLossConfig, {"reduction": "sum", "delta": 0.1}),
        (SmoothL1LossConfig, {"reduction": "sum", "beta": 0.0}),
        (
            MultiMarginLossConfig,
            {"reduction": "sum", "p": 1, "margin": -0.1, "weight": [1, 0, 2]},
        ),
    ],
)
def test_validation_pass(config, args):
    c = config(**args)
    for arg, value in args.items():
        assert getattr(c, arg) == value


@pytest.mark.parametrize(
    "name,config",
    [
        ("BCELoss", BCELossConfig),
        ("BCEWithLogitsLoss", BCEWithLogitsLossConfig),
        ("CrossEntropyLoss", CrossEntropyLossConfig),
        ("HuberLoss", HuberLossConfig),
        ("KLDivLoss", KLDivLossConfig),
        ("L1Loss", L1LossConfig),
        ("MSELoss", MSELossConfig),
        ("MultiMarginLoss", MultiMarginLossConfig),
        ("NLLLoss", NLLLossConfig),
        ("SmoothL1Loss", SmoothL1LossConfig),
    ],
)
def test_create_loss_config(name, config):
    assert create_loss_config(name) == config
