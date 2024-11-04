from typing import Optional, Sequence, Union

from pydantic import PositiveInt, computed_field

from clinicadl.utils.factories import DefaultFromLibrary

from .base import ImplementedNetwork, NetworkConfig
from .conv_decoder import ConvDecoderOptions
from .mlp import MLPOptions


class GeneratorConfig(NetworkConfig):
    """Config class for Generator."""

    latent_size: PositiveInt
    start_shape: Sequence[PositiveInt]
    conv_args: ConvDecoderOptions
    mlp_args: Union[Optional[MLPOptions], DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.GENERATOR
