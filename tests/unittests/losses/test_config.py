import pytest
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
)
from clinicadl.losses.enum import ImplementedLoss

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
    loss_from_config = c.get_object()
    assert isinstance(loss_from_config, loss)


def test_name():
    for name in ImplementedLoss:
        config = globals()[f"{name.value}Config"]
    c = config()
    assert c.name == name.value
