from typing import Dict, List, Optional, Tuple, Union

import torch.optim as optim
from pydantic import (
    NonNegativeFloat,
    PositiveFloat,
    computed_field,
    field_validator,
)

from clinicadl.utils.factories import DefaultFromLibrary

from .base import (
    ImplementedOptimizer,
    OptimizerConfig,
    _CapturableConfig,
    _EpsConfig,
    _FusedConfig,
    _MomentumConfig,
)

__all__ = [
    "AdadeltaConfig",
    "AdagradConfig",
    "AdamConfig",
    "RMSpropConfig",
    "SGDConfig",
]


class AdadeltaConfig(OptimizerConfig, _EpsConfig, _CapturableConfig):
    """
    Config class for :py:class:`torch.optim.Adadelta`.
    """

    rho: Union[NonNegativeFloat, Dict[str, NonNegativeFloat]]

    def __init__(
        self,
        *,
        lr: Union[PositiveFloat, Dict[str, PositiveFloat], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        freeze: Optional[Union[str, List[str]]] = None,
        rho: Union[
            NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
        ] = (DefaultFromLibrary.YES),
        eps: Union[
            NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
        ] = (DefaultFromLibrary.YES),
        weight_decay: Union[
            NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        foreach: Union[
            Optional[bool], Dict[str, Optional[bool]], DefaultFromLibrary
        ] = (DefaultFromLibrary.YES),
        capturable: Union[bool, Dict[str, bool], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        maximize: Union[
            bool, Dict[str, bool], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        differentiable: Union[bool, Dict[str, bool], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
    ):
        super().__init__(
            lr=lr,
            freeze=freeze,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
            foreach=foreach,
            capturable=capturable,
            maximize=maximize,
            differentiable=differentiable,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the optimizer."""
        return ImplementedOptimizer.ADADELTA.value

    def _get_class(self) -> type[optim.Optimizer]:
        """Returns the optimizer associated to this config class."""
        return optim.Adadelta

    @field_validator("rho")
    @classmethod
    def validator_rho(cls, v, ctx):
        return cls.validator_proba(v, ctx)


class AdagradConfig(OptimizerConfig, _EpsConfig, _FusedConfig):
    """
    Config class for :py:class:`torch.optim.Adagrad`.
    """

    lr_decay: Union[NonNegativeFloat, Dict[str, NonNegativeFloat]]
    initial_accumulator_value: Union[NonNegativeFloat, Dict[str, NonNegativeFloat]]

    def __init__(
        self,
        *,
        lr: Union[PositiveFloat, Dict[str, PositiveFloat], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        freeze: Optional[Union[str, List[str]]] = None,
        lr_decay: Union[
            NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        weight_decay: Union[
            NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        initial_accumulator_value: Union[
            NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        eps: Union[
            NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
        ] = (DefaultFromLibrary.YES),
        foreach: Union[
            Optional[bool], Dict[str, Optional[bool]], DefaultFromLibrary
        ] = (DefaultFromLibrary.YES),
        maximize: Union[
            bool, Dict[str, bool], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        differentiable: Union[bool, Dict[str, bool], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        fused: Union[Optional[bool], Dict[str, Optional[bool]], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
    ):
        super().__init__(
            lr=lr,
            freeze=freeze,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
            fused=fused,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the optimizer."""
        return ImplementedOptimizer.ADAGRAD.value

    def _get_class(self) -> type[optim.Optimizer]:
        """Returns the optimizer associated to this config class."""
        return optim.Adagrad


class AdamConfig(OptimizerConfig, _EpsConfig, _CapturableConfig, _FusedConfig):
    """
    Config class for :py:class:`torch.optim.Adam`.
    """

    betas: Union[
        Tuple[NonNegativeFloat, NonNegativeFloat],
        Dict[str, Tuple[NonNegativeFloat, NonNegativeFloat]],
    ]
    amsgrad: Union[bool, Dict[str, bool]]

    def __init__(
        self,
        *,
        lr: Union[PositiveFloat, Dict[str, PositiveFloat], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        freeze: Optional[Union[str, List[str]]] = None,
        betas: Union[
            Tuple[NonNegativeFloat, NonNegativeFloat],
            Dict[str, Tuple[NonNegativeFloat, NonNegativeFloat]],
            DefaultFromLibrary,
        ] = DefaultFromLibrary.YES,
        eps: Union[
            NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
        ] = (DefaultFromLibrary.YES),
        weight_decay: Union[
            NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        amsgrad: Union[
            bool, Dict[str, bool], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        foreach: Union[
            Optional[bool], Dict[str, Optional[bool]], DefaultFromLibrary
        ] = (DefaultFromLibrary.YES),
        maximize: Union[
            bool, Dict[str, bool], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        capturable: Union[bool, Dict[str, bool], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        differentiable: Union[bool, Dict[str, bool], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        fused: Union[Optional[bool], Dict[str, Optional[bool]], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
    ):
        super().__init__(
            lr=lr,
            freeze=freeze,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=foreach,
            capturable=capturable,
            maximize=maximize,
            differentiable=differentiable,
            fused=fused,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the optimizer."""
        return ImplementedOptimizer.ADAM.value

    def _get_class(self) -> type[optim.Optimizer]:
        """Returns the optimizer associated to this config class."""
        return optim.Adam

    @field_validator("betas")
    @classmethod
    def validator_betas(cls, v, ctx):
        return cls.validator_proba(v, ctx)


class RMSpropConfig(OptimizerConfig, _EpsConfig, _CapturableConfig, _MomentumConfig):
    """
    Config class for :py:class:`torch.optim.RMSprop`.
    """

    alpha: Union[NonNegativeFloat, Dict[str, NonNegativeFloat]]
    centered: Union[bool, Dict[str, bool]]

    def __init__(
        self,
        *,
        lr: Union[PositiveFloat, Dict[str, PositiveFloat], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        freeze: Optional[Union[str, List[str]]] = None,
        alpha: Union[
            NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
        ] = (DefaultFromLibrary.YES),
        eps: Union[
            NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
        ] = (DefaultFromLibrary.YES),
        weight_decay: Union[
            NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        momentum: Union[
            NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        centered: Union[
            bool, Dict[str, bool], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        capturable: Union[bool, Dict[str, bool], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        foreach: Union[
            Optional[bool], Dict[str, Optional[bool]], DefaultFromLibrary
        ] = (DefaultFromLibrary.YES),
        maximize: Union[
            bool, Dict[str, bool], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        differentiable: Union[bool, Dict[str, bool], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
    ):
        super().__init__(
            lr=lr,
            freeze=freeze,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
            capturable=capturable,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the optimizer."""
        return ImplementedOptimizer.RMS_PROP.value

    def _get_class(self) -> type[optim.Optimizer]:
        """Returns the optimizer associated to this config class."""
        return optim.RMSprop


class SGDConfig(OptimizerConfig, _FusedConfig, _MomentumConfig):
    """
    Config class for :py:class:`torch.optim.SGD`.
    """

    dampening: Union[float, Dict[str, float]]
    nesterov: Union[bool, Dict[str, bool]]

    def __init__(
        self,
        *,
        lr: Union[PositiveFloat, Dict[str, PositiveFloat], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        freeze: Optional[Union[str, List[str]]] = None,
        momentum: Union[
            NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        dampening: Union[
            float, Dict[str, float], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        weight_decay: Union[
            NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        nesterov: Union[
            bool, Dict[str, bool], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        maximize: Union[
            bool, Dict[str, bool], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        foreach: Union[
            Optional[bool], Dict[str, Optional[bool]], DefaultFromLibrary
        ] = (DefaultFromLibrary.YES),
        differentiable: Union[bool, Dict[str, bool], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        fused: Union[Optional[bool], Dict[str, Optional[bool]], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
    ):
        super().__init__(
            lr=lr,
            freeze=freeze,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            fused=fused,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the optimizer."""
        return ImplementedOptimizer.SGD.value

    def _get_class(self) -> type[optim.Optimizer]:
        """Returns the optimizer associated to this config class."""
        return optim.SGD
