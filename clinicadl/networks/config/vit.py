from typing import Optional, Sequence, Union

from pydantic import PositiveFloat, PositiveInt, model_validator

import clinicadl.networks.nn as nets
from clinicadl.networks.nn.layers.utils import ActivationParameters
from clinicadl.networks.nn.utils import ensure_tuple
from clinicadl.networks.nn.vit import (
    PosEmbedType,
    check_embedding_dim,
    check_patch_size,
)
from clinicadl.utils.factories import get_defaults_from

from .base import (
    NetworkConfig,
    _DropoutConfig,
    _InShapeConfig,
)

__all__ = ["ViTConfig", "ViTB16Config", "ViTB32Config", "ViTL16Config", "ViTL32Config"]

VIT_DEFAULTS = get_defaults_from(nets.ViT)
VIT_B_16_DEFAULTS = get_defaults_from(nets.ViTB16)
VIT_B_32_DEFAULTS = get_defaults_from(nets.ViTB32)
VIT_L_16_DEFAULTS = get_defaults_from(nets.ViTL16)
VIT_L_32_DEFAULTS = get_defaults_from(nets.ViTL32)


class ViTConfig(
    NetworkConfig,
    _InShapeConfig,
    _DropoutConfig,
):
    """
    Config class for :py:class:`clinicadl.networks.nn.ViT`.
    """

    in_shape: Sequence[PositiveInt]
    patch_size: Union[Sequence[PositiveInt], PositiveInt]
    num_outputs: Optional[PositiveInt]
    embedding_dim: PositiveInt = VIT_DEFAULTS["embedding_dim"]
    num_layers: PositiveInt = VIT_DEFAULTS["num_layers"]
    num_heads: PositiveInt = VIT_DEFAULTS["num_heads"]
    mlp_dim: PositiveInt = VIT_DEFAULTS["mlp_dim"]
    pos_embed_type: Optional[PosEmbedType] = VIT_DEFAULTS["pos_embed_type"]
    output_act: Optional[ActivationParameters] = VIT_DEFAULTS["output_act"]
    dropout: Optional[PositiveFloat] = VIT_DEFAULTS["dropout"]

    @model_validator(mode="after")
    def make_checks(self):
        _, *img_size = self.in_shape
        patch_size = ensure_tuple(self.patch_size, dim=len(img_size), name="patch_size")
        check_patch_size(patch_size, img_size)
        check_embedding_dim(self.embedding_dim, self.num_heads)

        return self


class ViTB16Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ViTB16`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = VIT_B_16_DEFAULTS["output_act"]
    pretrained: bool = VIT_B_16_DEFAULTS["pretrained"]


class ViTB32Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ViTB32`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = VIT_B_32_DEFAULTS["output_act"]
    pretrained: bool = VIT_B_32_DEFAULTS["pretrained"]


class ViTL16Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ViTL16`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = VIT_L_16_DEFAULTS["output_act"]
    pretrained: bool = VIT_L_16_DEFAULTS["pretrained"]


class ViTL32Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ViTL32`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = VIT_L_32_DEFAULTS["output_act"]
    pretrained: bool = VIT_L_32_DEFAULTS["pretrained"]
