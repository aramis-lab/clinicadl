from typing import Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt, computed_field

from clinicadl.monai_networks.nn.layers.utils import (
    ActivationParameters,
    NormalizationParameters,
)
from clinicadl.utils.factories import DefaultFromLibrary

from .base import ImplementedNetworks, NetworkConfig


class MLPOptions(BaseModel):
    """
    Config class for MLP when it is a submodule.
    See for example: :py:class:`clinicadl.monai_networks.nn.cnn.CNN`
    """

    hidden_channels: Sequence[PositiveInt]
    act: Union[
        Optional[ActivationParameters], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    output_act: Union[
        Optional[ActivationParameters], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    norm: Union[
        Optional[NormalizationParameters], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    dropout: Union[Optional[PositiveFloat], DefaultFromLibrary] = DefaultFromLibrary.YES
    bias: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES
    adn_ordering: Union[str, DefaultFromLibrary] = DefaultFromLibrary.YES

    # pydantic config
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        validate_default=True,
    )


class MLPConfig(NetworkConfig, MLPOptions):
    """Config class for Multi Layer Perceptron."""

    in_channels: PositiveInt
    out_channels: PositiveInt

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.MLP
