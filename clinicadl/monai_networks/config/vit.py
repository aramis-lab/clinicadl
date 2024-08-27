from enum import Enum
from typing import Optional, Tuple, Union

from pydantic import (
    NonNegativeFloat,
    PositiveInt,
    computed_field,
    field_validator,
    model_validator,
)

from clinicadl.utils.factories import DefaultFromLibrary

from .base import NetworkBaseConfig

__all__ = ["ViTConfig", "ViTAutoEncConfig"]


class PatchEmbeddingTypes(str, Enum):
    """Supported patch embedding types."""

    CONV = "conv"
    PERCEPTRON = "perceptron"


class PosEmbeddingTypes(str, Enum):
    """Supported positional embedding types."""

    NONE = "none"
    LEARNABLE = "learnable"
    SINCOS = "sincos"


class ClassificationActivation(str, Enum):
    """Supported activation layer for classification."""

    TANH = "Tanh"


class ViTConfig(NetworkBaseConfig):
    """Config class for ViT networks."""

    in_channels: PositiveInt
    img_size: Union[PositiveInt, Tuple[PositiveInt, ...]]
    patch_size: Union[PositiveInt, Tuple[PositiveInt, ...]]

    hidden_size: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    mlp_dim: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    num_layers: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    num_heads: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    proj_type: Union[PatchEmbeddingTypes, DefaultFromLibrary] = DefaultFromLibrary.YES
    pos_embed_type: Union[
        PosEmbeddingTypes, DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    classification: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES
    num_classes: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    dropout_rate: Union[NonNegativeFloat, DefaultFromLibrary] = DefaultFromLibrary.YES
    spatial_dims: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    post_activation: Union[
        Optional[ClassificationActivation], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    qkv_bias: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES
    save_attn: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    def dim(self) -> int:
        """Dimension of the images."""
        return self.spatial_dims

    @field_validator("dropout_rate")
    def validator_dropout(cls, v):
        """Checks that dropout is between 0 and 1."""
        return cls.base_validator_dropout(v)

    # TODO : add einops in dependencies
    # @model_validator(mode="before")
    # def check_einops(self):
    #     """Checks if the library einops is installed."""
    #     from importlib import util

    #     spec = util.find_spec("einops")
    #     if spec is None:
    #         raise ModuleNotFoundError("einops is not installed")
    #     return self

    @model_validator(mode="after")
    def model_validator(self):
        """Checks coherence between parameters."""
        assert self._check_dimensions(
            self.img_size
        ), f"You must passed an int or a sequence of {self.dim} ints (the dimensionality of your images) for img_size. You passed {self.img_size}."
        assert self._check_dimensions(
            self.patch_size
        ), f"You must passed an int or a sequence of {self.dim} ints (the dimensionality of your images) for patch_size. You passed {self.patch_size}."

        if (
            self.hidden_size != DefaultFromLibrary.YES
            and self.num_heads != DefaultFromLibrary.YES
        ):
            assert self._divide(
                self.hidden_size, self.num_heads
            ), f"hidden_size must be divisible by num_heads. You passed hidden_size={self.hidden_size} and num_heads={self.num_heads}."
        elif (
            self.hidden_size != DefaultFromLibrary.YES
            and self.num_heads == DefaultFromLibrary.YES
        ):
            raise ValueError("If you pass hidden_size, please also pass num_heads.")
        elif (
            self.hidden_size == DefaultFromLibrary.YES
            and self.num_heads != DefaultFromLibrary.YES
        ):
            raise ValueError("If you pass num_head, please also pass hidden_size.")

        return self

    def _divide(
        self,
        numerator: Union[int, Tuple[int, ...]],
        denominator: Union[int, Tuple[int, ...]],
    ) -> bool:
        """Checks if numerator is divisible by denominator."""
        if isinstance(numerator, int):
            numerator = (numerator,) * self.dim
        if isinstance(denominator, int):
            denominator = (denominator,) * self.dim
        for n, d in zip(numerator, denominator):
            if n % d != 0:
                return False
        return True


class ViTAutoEncConfig(ViTConfig):
    """Config class for ViT autoencoders."""

    out_channels: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    deconv_chns: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES

    @model_validator(mode="after")
    def model_validator_bis(self):
        """Checks coherence between parameters."""
        assert self._divide(
            self.img_size, self.patch_size
        ), f"img_size must be divisible by patch_size. You passed hidden_size={self.img_size} and num_heads={self.patch_size}."
        assert self._is_sqrt(
            self.patch_size
        ), f"patch_size must be square number(s). You passed {self.patch_size}."

        return self

    def _is_sqrt(self, value: Union[int, Tuple[int, ...]]) -> bool:
        """Checks if value is a square number."""
        import math

        if isinstance(value, int):
            value = (value,) * self.dim
        return all([int(math.sqrt(v)) == math.sqrt(v) for v in value])
