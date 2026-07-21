import pytest
import torch.nn as nn
from monai import losses as monai_losses
from pydantic import ValidationError

from clinicadl.losses.config import (
    BCELossConfig,
    BCEWithLogitsLossConfig,
    CrossEntropyLossConfig,
    DiceCELossConfig,
    DiceFocalLossConfig,
    DiceLossConfig,
    FocalLossConfig,
    GeneralizedDiceFocalLossConfig,
    GeneralizedDiceLossConfig,
    HuberLossConfig,
    ImplementedLoss,
    KLDivLossConfig,
    L1LossConfig,
    MSELossConfig,
    MultiMarginLossConfig,
    NLLLossConfig,
    SmoothL1LossConfig,
    SoftclDiceLossConfig,
    SSIMLossConfig,
    TverskyLossConfig,
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
    loss_from_config = c.get_object()
    assert isinstance(loss_from_config, loss)


def test_name():
    for name in ImplementedLoss:
        config = globals()[f"{name.value}Config"]
        c = config(spatial_dims=3)  # spatial_dims only for SSIMLossConfig
        assert c.name_ == name.value


@pytest.mark.parametrize(
    "config,loss,mandatory_args",
    [
        (DiceLossConfig, monai_losses.DiceLoss, {}),
        (DiceCELossConfig, monai_losses.DiceCELoss, {}),
        (DiceFocalLossConfig, monai_losses.DiceFocalLoss, {}),
        (GeneralizedDiceLossConfig, monai_losses.GeneralizedDiceLoss, {}),
        (
            GeneralizedDiceFocalLossConfig,
            monai_losses.GeneralizedDiceFocalLoss,
            {},
        ),
        (FocalLossConfig, monai_losses.FocalLoss, {}),
        (TverskyLossConfig, monai_losses.TverskyLoss, {}),
        (SoftclDiceLossConfig, monai_losses.SoftclDiceLoss, {}),
        (SSIMLossConfig, monai_losses.SSIMLoss, {"spatial_dims": 3}),
    ],
)
def test_get_monai_object(config, loss, mandatory_args):
    assert isinstance(config(**mandatory_args).get_object(), loss)


@pytest.mark.parametrize(
    "config",
    [
        DiceLossConfig,
        DiceCELossConfig,
        DiceFocalLossConfig,
        GeneralizedDiceLossConfig,
        GeneralizedDiceFocalLossConfig,
        TverskyLossConfig,
    ],
)
def test_monai_loss_rejects_multiple_activations(config):
    with pytest.raises(ValidationError, match="Only one"):
        config(sigmoid=True, softmax=True)


@pytest.mark.parametrize(
    "config,kwargs",
    [
        (DiceCELossConfig, {"label_smoothing": 1.1}),
        (DiceFocalLossConfig, {"alpha": 1.1}),
        (FocalLossConfig, {"alpha": 1.1}),
        (DiceFocalLossConfig, {"gamma": -1}),
        (GeneralizedDiceFocalLossConfig, {"lambda_gdl": -1}),
        (GeneralizedDiceLossConfig, {"w_type": "invalid"}),
        (TverskyLossConfig, {"smooth_dr": -1}),
        (SoftclDiceLossConfig, {"iter_": -1}),
        (SSIMLossConfig, {"spatial_dims": 1}),
    ],
)
def test_bad_monai_inputs(config, kwargs):
    with pytest.raises(ValidationError):
        config(**kwargs)


def test_monai_weight_list_is_converted_to_tensor():
    loss = FocalLossConfig(weight=[1, 2]).get_object()
    assert loss.class_weight is not None
    assert loss.class_weight.tolist() == [1, 2]
