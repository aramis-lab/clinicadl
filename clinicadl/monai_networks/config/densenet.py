from typing import Optional, Sequence, Union

from pydantic import PositiveFloat, PositiveInt, computed_field

from clinicadl.monai_networks.nn.layers.utils import ActivationParameters
from clinicadl.utils.factories import DefaultFromLibrary

from .base import ImplementedNetworks, NetworkConfig, PreTrainedConfig


class DenseNetConfig(NetworkConfig):
    """Config class for DenseNet."""

    spatial_dims: PositiveInt
    in_channels: PositiveInt
    num_outputs: Optional[PositiveInt]
    n_dense_layers: Union[
        Sequence[PositiveInt], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    init_features: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    growth_rate: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    bottleneck_factor: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    act: Union[ActivationParameters, DefaultFromLibrary] = DefaultFromLibrary.YES
    output_act: Union[
        Optional[ActivationParameters], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    dropout: Union[Optional[PositiveFloat], DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.DENSENET


class DenseNet121Config(PreTrainedConfig):
    """Config class for DenseNet-121."""

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.DENSENET_121


class DenseNet161Config(PreTrainedConfig):
    """Config class for DenseNet-161."""

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.DENSENET_161


class DenseNet169Config(PreTrainedConfig):
    """Config class for DenseNet-169."""

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.DENSENET_169


class DenseNet201Config(PreTrainedConfig):
    """Config class for DenseNet-201."""

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.DENSENET_201
