from typing import Optional, Sequence, Union

from pydantic import PositiveFloat, PositiveInt, model_validator

from clinicadl.networks.nn.layers.utils import ActivationParameters
from clinicadl.networks.nn.utils import ensure_tuple
from clinicadl.networks.nn.vit import (
    PosEmbedType,
    check_embedding_dim,
    check_patch_size,
)
from clinicadl.utils.factories import DefaultFromLibrary

from .base import (
    NetworkConfig,
    _DropOutConfig,
    _OptionalNumOutputsConfig,
    _OutputActConfig,
    _PretrainedFromLiteratureConfig,
)
from .cnns import _InShapeConfig


class ViTConfig(
    NetworkConfig,
    _InShapeConfig,
    _OptionalNumOutputsConfig,
    _OutputActConfig,
    _DropOutConfig,
):
    """
    Config class for :py:class:`clinicadl.networks.nn.ViT`.
    """

    patch_size: Union[Sequence[PositiveInt], PositiveInt]
    embedding_dim: PositiveInt
    num_layers: PositiveInt
    num_heads: PositiveInt
    mlp_dim: PositiveInt
    pos_embed_type: Optional[PosEmbedType]

    def __init__(
        self,
        in_shape: Sequence[PositiveInt],
        patch_size: Union[Sequence[PositiveInt], PositiveInt],
        num_outputs: Optional[PositiveInt],
        embedding_dim: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES,
        num_layers: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES,
        num_heads: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES,
        mlp_dim: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES,
        pos_embed_type: Union[Optional[PosEmbedType], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        output_act: Union[Optional[ActivationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        dropout: Union[
            Optional[PositiveFloat], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            in_shape=in_shape,
            patch_size=patch_size,
            num_outputs=num_outputs,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            pos_embed_type=pos_embed_type,
            output_act=output_act,
            dropout=dropout,
        )

    @model_validator(mode="after")
    def make_checks(self):
        _, *img_size = self.in_shape
        patch_size = ensure_tuple(self.patch_size, dim=len(img_size), name="patch_size")
        check_patch_size(patch_size, img_size)
        check_embedding_dim(self.embedding_dim, self.num_heads)

        return self


class ViTB16Config(_PretrainedFromLiteratureConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ViTB16`.
    """


class ViTB32Config(_PretrainedFromLiteratureConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ViTB32`.
    """


class ViTL16Config(_PretrainedFromLiteratureConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ViTL16`.
    """


class ViTL32Config(_PretrainedFromLiteratureConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ViTL32`.
    """
