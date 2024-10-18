from typing import Optional, Sequence, Union

from pydantic import PositiveInt, computed_field

from clinicadl.utils.factories import DefaultFromLibrary

from .base import ImplementedNetworks, NetworkConfig
from .conv_encoder import ConvEncoderOptions
from .mlp import MLPOptions


class CNNConfig(NetworkConfig):
    """Config class for CNN."""

    in_shape: Sequence[PositiveInt]
    num_outputs: PositiveInt
    conv_args: ConvEncoderOptions
    mlp_args: Union[Optional[MLPOptions], DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.CNN
