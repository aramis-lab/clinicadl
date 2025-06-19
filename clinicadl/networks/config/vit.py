from typing import Any, Callable, Optional, Sequence, Union

import torch.nn as nn
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

from ..nn.vit import ViT
from .base import (
    ImplementedNetwork,
    NetworkConfig,
    _DropOutConfig,
    _OptionalLastLinearLayersConfig,
    _OutputActConfig,
    _PreTrainedConfig,
)
from .cnns import _InShapeConfig

VIT_DEFAULTS = get_defaults_from(ViT)


class ViTConfig(
    NetworkConfig,
    _InShapeConfig,
    _OptionalLastLinearLayersConfig,
    _OutputActConfig,
    _DropOutConfig,
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


class _PreTrainedViTConfig(_PreTrainedConfig):
    """Base config class for SOTA ViTs."""

    @classmethod
    def _get_class(cls) -> Callable[[Any], nn.Module]:
        """Returns the network associated to this config class."""
        return nets.get_vit


class ViTB16Config(_PreTrainedViTConfig):
    """
    Config class for :py:func:`ViT-B/16 <clinicadl.networks.nn.get_vit>`.
    """

    @classmethod
    def _get_name(cls) -> str:
        """Returns the name of the class associated to this config class."""
        return ImplementedNetwork.VIT_B_16.value


class ViTB32Config(_PreTrainedViTConfig):
    """
    Config class for :py:func:`ViT-B/32 <clinicadl.networks.nn.get_vit>`.
    """

    @classmethod
    def _get_name(cls) -> str:
        """Returns the name of the class associated to this config class."""
        return ImplementedNetwork.VIT_B_32.value


class ViTL16Config(_PreTrainedViTConfig):
    """
    Config class for :py:func:`ViT-L/16 <clinicadl.networks.nn.get_vit>`.
    """

    @classmethod
    def _get_name(cls) -> str:
        """Returns the name of the class associated to this config class."""
        return ImplementedNetwork.VIT_L_16.value


class ViTL32Config(_PreTrainedViTConfig):
    """
    Config class for :py:func:`ViT-L/32 <clinicadl.networks.nn.get_vit>`.
    """

    @classmethod
    def _get_name(cls) -> str:
        """Returns the name of the class associated to this config class."""
        return ImplementedNetwork.VIT_L_32.value
