from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

from pydantic import PositiveInt, computed_field

from clinicadl.utils.factories import DefaultFromLibrary

from .regressor import RegressorConfig
from .utils.enum import ImplementedActFunctions, ImplementedNetworks

__all__ = ["ClassifierConfig", "DiscriminatorConfig", "CriticConfig"]


class ClassifierConfig(RegressorConfig):
    """Config class for classifiers."""

    classes: PositiveInt
    out_shape: Optional[Tuple[PositiveInt, ...]] = None
    last_act: Optional[
        Union[
            ImplementedActFunctions,
            Tuple[ImplementedActFunctions, Dict[str, Any]],
            DefaultFromLibrary,
        ]
    ] = DefaultFromLibrary.YES

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.CLASSIFIER


class DiscriminatorConfig(ClassifierConfig):
    """Config class for discriminators."""

    classes: Optional[PositiveInt] = None

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.DISCRIMINATOR


class CriticConfig(ClassifierConfig):
    """Config class for discriminators."""

    classes: Optional[PositiveInt] = None
    last_act: Optional[
        Union[
            ImplementedActFunctions,
            Tuple[ImplementedActFunctions, Dict[str, Any]],
            DefaultFromLibrary,
        ]
    ] = None

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.CRITIC
