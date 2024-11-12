from typing import Optional, Sequence, Union

from pydantic import PositiveInt, computed_field, model_validator

from clinicadl.networks.nn.utils import ensure_tuple
from clinicadl.networks.nn.vit import (
    PosEmbedType,
    check_embedding_dim,
    check_patch_size,
)
from clinicadl.utils.factories import DefaultFromLibrary

from .base import (
    ImplementedNetwork,
    NetworkConfig,
    NetworkType,
    _DropOutConfig,
    _OptionalLastLinearLayersConfig,
    _OutputActConfig,
    _PreTrainedConfig,
)
from .cnns import _InShapeConfig


class ViTConfig(
    NetworkConfig,
    _InShapeConfig,
    _OptionalLastLinearLayersConfig,
    _OutputActConfig,
    _DropOutConfig,
):
    """Config class for ViT networks."""

    patch_size: Union[Sequence[PositiveInt], PositiveInt]
    embedding_dim: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    num_layers: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    num_heads: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    mlp_dim: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    pos_embed_type: Union[
        Optional[PosEmbedType], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.VIT

    @model_validator(mode="after")
    def make_checks(self):
        _, *img_size = self.in_shape
        patch_size = ensure_tuple(self.patch_size, dim=len(img_size), name="patch_size")
        check_patch_size(patch_size, img_size)
        if (
            self.embedding_dim != DefaultFromLibrary.YES
            and self.num_heads != DefaultFromLibrary.YES
        ):
            check_embedding_dim(self.embedding_dim, self.num_heads)

        return self


class _PreTrainedViTConfig(_PreTrainedConfig):
    """Base config class for SOTA ResNets."""

    @computed_field
    @property
    def _type(self) -> NetworkType:
        """To know where to look for the network."""
        return NetworkType.VIT


class ViTB16Config(_PreTrainedViTConfig):
    """Config class for ViT-B/16."""

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.VIT_B_16


class ViTB32Config(_PreTrainedViTConfig):
    """Config class for ViT-B/32."""

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.VIT_B_32


class ViTL16Config(_PreTrainedViTConfig):
    """Config class for ViT-L/16."""

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.VIT_L_16


class ViTL32Config(_PreTrainedViTConfig):
    """Config class for ViT-L/32."""

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.VIT_L_32
