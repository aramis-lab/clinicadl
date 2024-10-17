from typing import Optional, Sequence, Union

from pydantic import PositiveFloat, PositiveInt, computed_field

from clinicadl.monai_networks.nn.layers.utils import ActivationParameters
from clinicadl.monai_networks.nn.vit import PosEmbedType
from clinicadl.utils.factories import DefaultFromLibrary

from .base import ImplementedNetworks, NetworkConfig, PreTrainedConfig


class ViTConfig(NetworkConfig):
    """Config class for ViT networks."""

    in_shape: Sequence[PositiveInt]
    patch_size: Union[Sequence[PositiveInt], PositiveInt]
    num_outputs: Optional[PositiveInt]
    embedding_dim: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    num_layers: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    num_heads: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    mlp_dim: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    pos_embed_type: Union[
        Optional[Union[str, PosEmbedType]], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    output_act: Union[
        Optional[ActivationParameters], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    dropout: Union[Optional[PositiveFloat], DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.VIT


class ViTB16Config(PreTrainedConfig):
    """Config class for ViT-B/16."""

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.VIT_B_16


class ViTB32Config(PreTrainedConfig):
    """Config class for ViT-B/32."""

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.VIT_B_32


class ViTL16Config(PreTrainedConfig):
    """Config class for ViT-L/16."""

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.VIT_L_16


class ViTL32Config(PreTrainedConfig):
    """Config class for ViT-L/32."""

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.VIT_L_32
