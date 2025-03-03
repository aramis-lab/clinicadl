from typing import Any, List, Optional, Union

import torch
from pydantic import (
    NonNegativeFloat,
    PositiveFloat,
    computed_field,
    field_validator,
)

from clinicadl.utils.config import ClinicaDLConfig, NewClinicaDLConfig
from clinicadl.utils.factories import DefaultFromLibrary

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


class LossConfig(NewClinicaDLConfig):
    """Base config class for the loss function."""

    reduction: Reduction

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


class _WeightConfig(ClinicaDLConfig):
    """Base config class for loss functions with 'weight' argument."""

    weight: Optional[List[NonNegativeFloat]]


class NLLLossConfig(LossConfig, _WeightConfig):
    """
    Config class for :py:class:`torch.nn.NLLLoss`.
    """

    ignore_index: int

    def __init__(
        self,
        weight: Union[
            Optional[List[NonNegativeFloat]], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        ignore_index: Union[int, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(weight=weight, ignore_index=ignore_index, reduction=reduction)

    @computed_field
    @property
    def name(self) -> str:
        """The name of the loss."""
        return ImplementedLoss.NLL.value

    def _get_class(self) -> type[torch.nn.Module]:
        """Returns the loss function associated to this config class."""
        return torch.nn.NLLLoss

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

    label_smoothing: NonNegativeFloat

    def __init__(
        self,
        weight: Union[
            Optional[List[NonNegativeFloat]], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        ignore_index: Union[int, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        label_smoothing: Union[NonNegativeFloat, DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
    ):
        super(NLLLossConfig, self).__init__(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the loss."""
        return ImplementedLoss.CROSS_ENTROPY.value

    def _get_class(self) -> type[torch.nn.Module]:
        """Returns the loss function associated to this config class."""
        return torch.nn.CrossEntropyLoss

    @field_validator("label_smoothing")
    @classmethod
    def validator_label_smoothing(cls, v):
        if isinstance(v, float):
            assert (
                0 <= v <= 1
            ), f"label_smoothing must be between 0 and 1 but it has been set to {v}."
        return v


class BCELossConfig(LossConfig, _WeightConfig):
    """
    Config class for :py:class:`torch.nn.BCELoss`.
    """

    def __init__(
        self,
        weight: Union[
            Optional[List[NonNegativeFloat]], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            weight=weight,
            reduction=reduction,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the loss."""
        return ImplementedLoss.BCE.value

    def _get_class(self) -> type[torch.nn.Module]:
        """Returns the loss function associated to this config class."""
        return torch.nn.BCELoss

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

    pos_weight: Optional[List[Any]]

    def __init__(
        self,
        weight: Union[
            Optional[List[NonNegativeFloat]], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        pos_weight: Union[
            Optional[List[Any]], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
    ):
        super(BCELossConfig, self).__init__(
            weight=weight,
            reduction=reduction,
            pos_weight=pos_weight,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the loss."""
        return ImplementedLoss.BCE_LOGITS.value

    def _get_class(self) -> type[torch.nn.Module]:
        """Returns the loss function associated to this config class."""
        return torch.nn.BCEWithLogitsLoss

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


class MultiMarginLossConfig(LossConfig, _WeightConfig):
    """
    Config class for :py:class:`torch.nn.MultiMarginLoss`.
    """

    p: Order
    margin: float

    def __init__(
        self,
        p: Union[Order, DefaultFromLibrary] = DefaultFromLibrary.YES,
        margin: Union[float, DefaultFromLibrary] = DefaultFromLibrary.YES,
        weight: Union[
            Optional[List[NonNegativeFloat]], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            p=p,
            margin=margin,
            weight=weight,
            reduction=reduction,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the loss."""
        return ImplementedLoss.MULTI_MARGIN.value

    def _get_class(self) -> type[torch.nn.Module]:
        """Returns the loss function associated to this config class."""
        return torch.nn.MultiMarginLoss


class KLDivLossConfig(LossConfig):
    """
    Config class for :py:class:`torch.nn.KLDivLoss`.
    """

    log_target: bool

    def __init__(
        self,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        log_target: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            reduction=reduction,
            log_target=log_target,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the loss."""
        return ImplementedLoss.KLDIV.value

    def _get_class(self) -> type[torch.nn.Module]:
        """Returns the loss function associated to this config class."""
        return torch.nn.KLDivLoss


class HuberLossConfig(LossConfig):
    """
    Config class for :py:class:`torch.nn.HuberLoss`.
    """

    delta: PositiveFloat

    def __init__(
        self,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        delta: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            reduction=reduction,
            delta=delta,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the loss."""
        return ImplementedLoss.HUBER.value

    def _get_class(self) -> type[torch.nn.Module]:
        """Returns the loss function associated to this config class."""
        return torch.nn.HuberLoss


class SmoothL1LossConfig(LossConfig):
    """
    Config class for :py:class:`torch.nn.SmoothL1Loss`.
    """

    beta: NonNegativeFloat

    def __init__(
        self,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        beta: Union[NonNegativeFloat, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            reduction=reduction,
            beta=beta,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the loss."""
        return ImplementedLoss.SMOOTH_L1.value

    def _get_class(self) -> type[torch.nn.Module]:
        """Returns the loss function associated to this config class."""
        return torch.nn.SmoothL1Loss


class L1LossConfig(LossConfig):
    """
    Config class for :py:class:`torch.nn.L1Loss`.
    """

    def __init__(
        self,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            reduction=reduction,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the loss."""
        return ImplementedLoss.L1.value

    def _get_class(self) -> type[torch.nn.Module]:
        """Returns the loss function associated to this config class."""
        return torch.nn.L1Loss


class MSELossConfig(LossConfig):
    """
    Config class for :py:class:`torch.nn.MSELoss`.
    """

    def __init__(
        self,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            reduction=reduction,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the loss."""
        return ImplementedLoss.MSE.value

    def _get_class(self) -> type[torch.nn.Module]:
        """Returns the loss function associated to this config class."""
        return torch.nn.MSELoss


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
    transform = ImplementedLoss(name)
    config_name = "".join([transform, "Config"])
    config = globals()[config_name]

    return config(**kwargs)
