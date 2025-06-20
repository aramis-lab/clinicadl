"""Config classes for loss functions natively supported in ``ClinicaDL``. Based on
:torch:`PyTorch loss functions <nn.html#loss-functions>`."""

from typing import Any, List, Optional, Union

import torch
from pydantic import (
    NonNegativeFloat,
    PositiveFloat,
    field_validator,
)

from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.factories import get_defaults_from

from .enum import ImplementedLoss, Order, Reduction

__all__ = [
    "LossConfig",
    "NLLLossConfig",
    "CrossEntropyLossConfig",
    "BCELossConfig",
    "BCEWithLogitsLossConfig",
    "MultiMarginLossConfig",
    "KLDivLossConfig",
    "HuberLossConfig",
    "SmoothL1LossConfig",
    "L1LossConfig",
    "MSELossConfig",
    "get_loss_function_config",
]

NLL_TORCH_DEFAULTS = get_defaults_from(torch.nn.NLLLoss)
CROSS_ENTROPY_TORCH_DEFAULTS = get_defaults_from(torch.nn.CrossEntropyLoss)
BCE_TORCH_DEFAULTS = get_defaults_from(torch.nn.BCELoss)
BCE_LOGITS_TORCH_DEFAULTS = get_defaults_from(torch.nn.BCEWithLogitsLoss)
MULTI_MARGIN_LOSS_TORCH_DEFAULTS = get_defaults_from(torch.nn.MultiMarginLoss)
KL_DIV_LOSS_TORCH_DEFAULTS = get_defaults_from(torch.nn.KLDivLoss)
HUBER_LOSS_TORCH_DEFAULTS = get_defaults_from(torch.nn.HuberLoss)
SMOOTH_L1_LOSS_TORCH_DEFAULTS = get_defaults_from(torch.nn.SmoothL1Loss)
L1_TORCH_DEFAULT = get_defaults_from(torch.nn.L1Loss)
MSE_TORCH_DEFAULT = get_defaults_from(torch.nn.MSELoss)


class LossConfig(ObjectConfig):
    """Base config class for the loss function."""

    def get_object(self) -> torch.nn.Module:
        """
        Returns the loss function associated to this configuration,
        parametrized with the parameters passed by the user.

        Returns
        -------
        torch.nn.Module:
            The PyTorch loss function.
        """
        params = self.model_dump(exclude="name")
        if "weight" in params and params["weight"]:
            params["weight"] = torch.Tensor(params["weight"])
        if "pos_weight" in params and params["pos_weight"]:
            params["pos_weight"] = torch.Tensor(params["pos_weight"])

        associated_class = self._get_class()

        return associated_class(**params)

    @classmethod
    def _get_class(cls) -> type[torch.nn.Module]:
        """Returns the loss function associated to this config class."""
        return getattr(torch.nn, cls._get_name())


class NLLLossConfig(LossConfig):
    """
    Config class for :py:class:`torch.nn.NLLLoss`.
    """

    weight: Optional[List[NonNegativeFloat]] = NLL_TORCH_DEFAULTS["weight"]
    ignore_index: int = NLL_TORCH_DEFAULTS["ignore_index"]
    reduction: Reduction = NLL_TORCH_DEFAULTS["reduction"]

    @field_validator("ignore_index")
    @classmethod
    def validator_ignore_index(cls, v):
        if isinstance(v, int):
            assert (
                v == -100 or 0 <= v
            ), "ignore_index must be a positive int (or -100 when disabled)."
        return v


class CrossEntropyLossConfig(NLLLossConfig):
    """
    Config class for :py:class:`torch.nn.CrossEntropyLoss`.
    """

    weight: Optional[List[NonNegativeFloat]] = CROSS_ENTROPY_TORCH_DEFAULTS["weight"]
    ignore_index: int = CROSS_ENTROPY_TORCH_DEFAULTS["ignore_index"]
    reduction: Reduction = CROSS_ENTROPY_TORCH_DEFAULTS["reduction"]
    label_smoothing: NonNegativeFloat = CROSS_ENTROPY_TORCH_DEFAULTS["label_smoothing"]

    @field_validator("label_smoothing")
    @classmethod
    def validator_label_smoothing(cls, v):
        if isinstance(v, float):
            assert (
                0 <= v <= 1
            ), f"label_smoothing must be between 0 and 1 but it has been set to {v}."
        return v


class BCELossConfig(LossConfig):
    """
    Config class for :py:class:`torch.nn.BCELoss`.
    """

    weight: Optional[List[NonNegativeFloat]] = BCE_TORCH_DEFAULTS["weight"]
    reduction: Reduction = BCE_TORCH_DEFAULTS["reduction"]

    @field_validator("weight")
    @classmethod
    def validator_weight(cls, v):
        if v is not None:
            raise ValueError(
                "'weight' with BCEWithLogitsLoss is not supported by ClinicaDL currently. Please leave it to None."
            )
        return v


class BCEWithLogitsLossConfig(BCELossConfig):
    """
    Config class for :py:class:`torch.nn.BCEWithLogitsLoss`.
    """

    weight: Optional[List[NonNegativeFloat]] = BCE_LOGITS_TORCH_DEFAULTS["weight"]
    reduction: Reduction = BCE_LOGITS_TORCH_DEFAULTS["reduction"]
    pos_weight: Optional[List[Any]] = BCE_LOGITS_TORCH_DEFAULTS["pos_weight"]

    @field_validator("pos_weight")
    @classmethod
    def validator_pos_weight(cls, v):
        if isinstance(v, list):
            check = cls._recursive_float_check(v)
            if not check:
                raise ValueError(
                    f"elements in pos_weight must be non-negative float, got: {v}"
                )
        return v

    @classmethod
    def _recursive_float_check(cls, item):
        if isinstance(item, list):
            return all(cls._recursive_float_check(i) for i in item)
        else:
            return (isinstance(item, float) or isinstance(item, int)) and item >= 0


class MultiMarginLossConfig(LossConfig):
    """
    Config class for :py:class:`torch.nn.MultiMarginLoss`.
    """

    p: Order = MULTI_MARGIN_LOSS_TORCH_DEFAULTS["p"]
    margin: float = MULTI_MARGIN_LOSS_TORCH_DEFAULTS["margin"]
    weight: Optional[List[NonNegativeFloat]] = MULTI_MARGIN_LOSS_TORCH_DEFAULTS[
        "weight"
    ]
    reduction: Reduction = MULTI_MARGIN_LOSS_TORCH_DEFAULTS["reduction"]


class KLDivLossConfig(LossConfig):
    """
    Config class for :py:class:`torch.nn.KLDivLoss`.
    """

    reduction: Reduction = KL_DIV_LOSS_TORCH_DEFAULTS["reduction"]
    log_target: bool = KL_DIV_LOSS_TORCH_DEFAULTS["log_target"]


class HuberLossConfig(LossConfig):
    """
    Config class for :py:class:`torch.nn.HuberLoss`.
    """

    reduction: Reduction = HUBER_LOSS_TORCH_DEFAULTS["reduction"]
    delta: PositiveFloat = HUBER_LOSS_TORCH_DEFAULTS["delta"]


class SmoothL1LossConfig(LossConfig):
    """
    Config class for :py:class:`torch.nn.SmoothL1Loss`.
    """

    reduction: Reduction = SMOOTH_L1_LOSS_TORCH_DEFAULTS["reduction"]
    beta: NonNegativeFloat = SMOOTH_L1_LOSS_TORCH_DEFAULTS["beta"]


class L1LossConfig(LossConfig):
    """
    Config class for :py:class:`torch.nn.L1Loss`.
    """

    reduction: Reduction = L1_TORCH_DEFAULT["reduction"]


class MSELossConfig(LossConfig):
    """
    Config class for :py:class:`torch.nn.MSELoss`.
    """

    reduction: Reduction = MSE_TORCH_DEFAULT["reduction"]


def get_loss_function_config(
    name: Union[str, ImplementedLoss], **kwargs: Any
) -> LossConfig:
    """
    Factory function to get a loss function configuration object from its name
    and parameters.

    Parameters
    ----------
    name : Union[str, ImplementedLoss]
        the name of the loss function. Check our documentation to know
        available losses.
    **kwargs : Any
        any parameter of the loss function. Check our documentation on losses to
        know these parameters.

    Returns
    -------
    LossConfig
        the config object. Default values will be returned for the parameters
        not passed by the user.
    """
    loss = ImplementedLoss(name).value
    config_name = f"{loss}Config"
    config = globals()[config_name]

    return config(**kwargs)
