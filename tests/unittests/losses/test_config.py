import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

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
    get_loss_function_config,
)

BAD_INPUTS = [
    (
        {"reduction": "none"},
        [
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
        ],
    ),
    (
        {"reduction": None},
        [
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
        ],
    ),
    (
        {"weight": [1, -1, 2]},
        [
            NLLLossConfig,
            CrossEntropyLossConfig,
            BCELossConfig,
            BCEWithLogitsLossConfig,
            MultiMarginLossConfig,
        ],
    ),
    (
        {"weight": [1, 1, 2]},
        [
            BCELossConfig,
            BCEWithLogitsLossConfig,
        ],
    ),
    ({"ignore_index": -1}, [NLLLossConfig, CrossEntropyLossConfig]),
    ({"label_smoothing": 1.1}, CrossEntropyLossConfig),
    ({"pos_weight": [1, -1, 2]}, BCEWithLogitsLossConfig),
    ({"delta": 0.0}, HuberLossConfig),
    ({"beta": -0.1}, SmoothL1LossConfig),
    ({"p": 3}, MultiMarginLossConfig),
    ({"margin": None}, MultiMarginLossConfig),
    ({"log_target": None}, KLDivLossConfig),
]

GOOD_INPUTS = [
    (
        {"reduction": "mean"},
        [
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
        ],
    ),
    (
        {"reduction": "sum"},
        [
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
        ],
    ),
    (
        {"weight": [1, 1, 2]},
        [
            NLLLossConfig,
            CrossEntropyLossConfig,
            MultiMarginLossConfig,
        ],
    ),
    ({"ignore_index": -100}, [NLLLossConfig, CrossEntropyLossConfig]),
    ({"ignore_index": 0}, [NLLLossConfig, CrossEntropyLossConfig]),
    ({"label_smoothing": 0.5}, CrossEntropyLossConfig),
    ({"pos_weight": [1, 1, 2]}, BCEWithLogitsLossConfig),
    ({"delta": 0.1}, HuberLossConfig),
    ({"beta": 0}, SmoothL1LossConfig),
    ({"p": 1}, MultiMarginLossConfig),
    ({"p": 2}, MultiMarginLossConfig),
    ({"margin": -0.5}, MultiMarginLossConfig),
    ({"log_target": True}, KLDivLossConfig),
    ({"log_target": False}, KLDivLossConfig),
]


@pytest.mark.parametrize("args,configs", BAD_INPUTS)
def test_bad_inputs(args, configs):
    if not isinstance(configs, list):
        configs = [configs]
    for config in configs:
        with pytest.raises(ValidationError):
            config(**args)


@pytest.mark.parametrize("args,configs", GOOD_INPUTS)
def test_good_inputs(args: dict, configs):
    if not isinstance(configs, list):
        configs = [configs]
    for config in configs:
        c = config(**args)
        for arg, value in args.items():
            assert getattr(c, arg) == value


@pytest.mark.parametrize(
    "config,loss",
    [
        (BCELossConfig, nn.BCELoss),
        (BCEWithLogitsLossConfig, nn.BCEWithLogitsLoss),
        (CrossEntropyLossConfig, nn.CrossEntropyLoss),
        (HuberLossConfig, nn.HuberLoss),
        (KLDivLossConfig, nn.KLDivLoss),
        (L1LossConfig, nn.L1Loss),
        (MSELossConfig, nn.MSELoss),
        (MultiMarginLossConfig, nn.MultiMarginLoss),
        (NLLLossConfig, nn.NLLLoss),
        (SmoothL1LossConfig, nn.SmoothL1Loss),
    ],
)
def test_get_object(config, loss):
    c = config()
    transform_from_config = c.get_object()
    assert isinstance(transform_from_config, loss)


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
def test_get_transform_config(name, config):
    c = get_loss_function_config(name)
    assert c.name == name
    assert isinstance(c, config)
    with pytest.raises(ValueError):
        get_loss_function_config("abc")

    if name == "NLLLoss":
        config = get_loss_function_config("NLLLoss", weight=[1, 2])
        assert config.name == "NLLLoss"
        assert config.weight == [1, 2]
        assert config.reduction == "mean"

        assert (config.get_object().weight == torch.Tensor([1, 2])).all()
    elif name == "BCEWithLogitsLoss":
        config = get_loss_function_config("BCEWithLogitsLoss", pos_weight=[1, 2])
        assert config.name == "BCEWithLogitsLoss"
        assert config.pos_weight == [1, 2]
        assert config.reduction == "mean"
        assert (config.get_object().pos_weight == torch.Tensor([1, 2])).all()
