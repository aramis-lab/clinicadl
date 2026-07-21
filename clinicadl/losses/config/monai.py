"""Config classes for commonly used MONAI loss functions."""

from typing import Callable, Optional, Union

import monai.losses
import torch
from pydantic import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from clinicadl.metrics.config.enum import Kernel
from clinicadl.metrics.config.reconstruction import BaseSSIMConfig
from clinicadl.utils.factories import get_defaults_from

from .configs import LossConfig
from .enum import GeneralizedDiceWeight, Reduction

__all__ = [
    "MonaiLossConfig",
    "DiceLossConfig",
    "DiceCELossConfig",
    "DiceFocalLossConfig",
    "GeneralizedDiceLossConfig",
    "GeneralizedDiceFocalLossConfig",
    "FocalLossConfig",
    "TverskyLossConfig",
    "SoftclDiceLossConfig",
    "SSIMLossConfig",
]

DICE_MONAI_DEFAULTS = get_defaults_from(monai.losses.DiceLoss)
DICE_CE_MONAI_DEFAULTS = get_defaults_from(monai.losses.DiceCELoss)
DICE_FOCAL_MONAI_DEFAULTS = get_defaults_from(monai.losses.DiceFocalLoss)
GENERALIZED_DICE_MONAI_DEFAULTS = get_defaults_from(monai.losses.GeneralizedDiceLoss)
GENERALIZED_DICE_FOCAL_MONAI_DEFAULTS = get_defaults_from(
    monai.losses.GeneralizedDiceFocalLoss
)
FOCAL_MONAI_DEFAULTS = get_defaults_from(monai.losses.FocalLoss)
TVERSKY_MONAI_DEFAULTS = get_defaults_from(monai.losses.TverskyLoss)
SOFT_CL_DICE_MONAI_DEFAULTS = get_defaults_from(monai.losses.SoftclDiceLoss)
SSIM_MONAI_DEFAULTS = get_defaults_from(monai.losses.SSIMLoss)

SerializableWeight = Optional[Union[NonNegativeFloat, list[NonNegativeFloat]]]


class MonaiLossConfig(LossConfig):
    """Base config class for MONAI loss functions."""

    @classmethod
    def _get_class(cls) -> type[torch.nn.Module]:
        """Returns the MONAI loss function associated to this config class."""
        return getattr(monai.losses, cls._get_name())


class _OverlapLossConfig(MonaiLossConfig):
    """Parameters shared by overlap-based MONAI losses."""

    include_background: bool = DICE_MONAI_DEFAULTS["include_background"]
    to_onehot_y: bool = DICE_MONAI_DEFAULTS["to_onehot_y"]
    sigmoid: bool = DICE_MONAI_DEFAULTS["sigmoid"]
    softmax: bool = DICE_MONAI_DEFAULTS["softmax"]
    other_act: Optional[Callable[[torch.Tensor], torch.Tensor]] = DICE_MONAI_DEFAULTS[
        "other_act"
    ]
    reduction: Reduction = DICE_MONAI_DEFAULTS["reduction"]
    smooth_nr: NonNegativeFloat = DICE_MONAI_DEFAULTS["smooth_nr"]
    smooth_dr: NonNegativeFloat = DICE_MONAI_DEFAULTS["smooth_dr"]
    batch: bool = DICE_MONAI_DEFAULTS["batch"]

    @model_validator(mode="after")
    def validate_activation(self):
        """Only one activation may be enabled at a time."""
        if sum((self.sigmoid, self.softmax, self.other_act is not None)) > 1:
            raise ValueError(
                "Only one of 'sigmoid', 'softmax' and 'other_act' may be set."
            )
        return self


class _DiceLossConfig(_OverlapLossConfig):
    """Parameters shared by Dice-based MONAI losses."""

    squared_pred: bool = DICE_MONAI_DEFAULTS["squared_pred"]
    jaccard: bool = DICE_MONAI_DEFAULTS["jaccard"]


class DiceLossConfig(_DiceLossConfig):
    """Config class for :py:class:`monai.losses.DiceLoss`."""

    weight: SerializableWeight = DICE_MONAI_DEFAULTS["weight"]
    soft_label: bool = DICE_MONAI_DEFAULTS["soft_label"]


class DiceCELossConfig(_DiceLossConfig):
    """Config class for :py:class:`monai.losses.DiceCELoss`."""

    weight: Optional[list[NonNegativeFloat]] = DICE_CE_MONAI_DEFAULTS["weight"]
    lambda_dice: NonNegativeFloat = DICE_CE_MONAI_DEFAULTS["lambda_dice"]
    lambda_ce: NonNegativeFloat = DICE_CE_MONAI_DEFAULTS["lambda_ce"]
    label_smoothing: NonNegativeFloat = DICE_CE_MONAI_DEFAULTS["label_smoothing"]

    @field_validator("label_smoothing")
    @classmethod
    def validate_label_smoothing(cls, value):
        if value > 1:
            raise ValueError("'label_smoothing' must be between 0 and 1.")
        return value


class DiceFocalLossConfig(_DiceLossConfig):
    """Config class for :py:class:`monai.losses.DiceFocalLoss`."""

    gamma: NonNegativeFloat = DICE_FOCAL_MONAI_DEFAULTS["gamma"]
    weight: SerializableWeight = DICE_FOCAL_MONAI_DEFAULTS["weight"]
    lambda_dice: NonNegativeFloat = DICE_FOCAL_MONAI_DEFAULTS["lambda_dice"]
    lambda_focal: NonNegativeFloat = DICE_FOCAL_MONAI_DEFAULTS["lambda_focal"]
    alpha: Optional[NonNegativeFloat] = DICE_FOCAL_MONAI_DEFAULTS["alpha"]

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, value):
        if value is not None and value > 1:
            raise ValueError("'alpha' must be between 0 and 1.")
        return value


class _GeneralizedDiceLossConfig(_OverlapLossConfig):
    """Parameters shared by generalized Dice MONAI losses."""

    w_type: GeneralizedDiceWeight = GENERALIZED_DICE_MONAI_DEFAULTS["w_type"]


class GeneralizedDiceLossConfig(_GeneralizedDiceLossConfig):
    """Config class for :py:class:`monai.losses.GeneralizedDiceLoss`."""

    soft_label: bool = GENERALIZED_DICE_MONAI_DEFAULTS["soft_label"]


class GeneralizedDiceFocalLossConfig(_GeneralizedDiceLossConfig):
    """Config class for :py:class:`monai.losses.GeneralizedDiceFocalLoss`."""

    gamma: NonNegativeFloat = GENERALIZED_DICE_FOCAL_MONAI_DEFAULTS["gamma"]
    weight: SerializableWeight = GENERALIZED_DICE_FOCAL_MONAI_DEFAULTS["weight"]
    lambda_gdl: NonNegativeFloat = GENERALIZED_DICE_FOCAL_MONAI_DEFAULTS["lambda_gdl"]
    lambda_focal: NonNegativeFloat = GENERALIZED_DICE_FOCAL_MONAI_DEFAULTS[
        "lambda_focal"
    ]


class FocalLossConfig(MonaiLossConfig):
    """Config class for :py:class:`monai.losses.FocalLoss`."""

    include_background: bool = FOCAL_MONAI_DEFAULTS["include_background"]
    to_onehot_y: bool = FOCAL_MONAI_DEFAULTS["to_onehot_y"]
    gamma: NonNegativeFloat = FOCAL_MONAI_DEFAULTS["gamma"]
    alpha: Optional[NonNegativeFloat] = FOCAL_MONAI_DEFAULTS["alpha"]
    weight: SerializableWeight = FOCAL_MONAI_DEFAULTS["weight"]
    reduction: Reduction = FOCAL_MONAI_DEFAULTS["reduction"]
    use_softmax: bool = FOCAL_MONAI_DEFAULTS["use_softmax"]

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, value):
        if value is not None and value > 1:
            raise ValueError("'alpha' must be between 0 and 1.")
        return value


class TverskyLossConfig(_OverlapLossConfig):
    """Config class for :py:class:`monai.losses.TverskyLoss`."""

    alpha: NonNegativeFloat = TVERSKY_MONAI_DEFAULTS["alpha"]
    beta: NonNegativeFloat = TVERSKY_MONAI_DEFAULTS["beta"]
    soft_label: bool = TVERSKY_MONAI_DEFAULTS["soft_label"]


class SoftclDiceLossConfig(MonaiLossConfig):
    """Config class for :py:class:`monai.losses.SoftclDiceLoss`."""

    iter_: NonNegativeInt = SOFT_CL_DICE_MONAI_DEFAULTS["iter_"]
    smooth: float = SOFT_CL_DICE_MONAI_DEFAULTS["smooth"]


class SSIMLossConfig(MonaiLossConfig, BaseSSIMConfig):
    """Config class for :py:class:`monai.losses.ssim_loss.SSIMLoss`."""

    spatial_dims: PositiveInt
    data_range: PositiveFloat = SSIM_MONAI_DEFAULTS["data_range"]
    kernel_type: Kernel = SSIM_MONAI_DEFAULTS["kernel_type"]
    win_size: Union[PositiveInt, tuple[PositiveInt, ...]] = SSIM_MONAI_DEFAULTS[
        "win_size"
    ]
    kernel_sigma: Union[PositiveFloat, tuple[PositiveFloat, ...]] = SSIM_MONAI_DEFAULTS[
        "kernel_sigma"
    ]
    k1: NonNegativeFloat = SSIM_MONAI_DEFAULTS["k1"]
    k2: NonNegativeFloat = SSIM_MONAI_DEFAULTS["k2"]
    reduction: Reduction = SSIM_MONAI_DEFAULTS["reduction"]
