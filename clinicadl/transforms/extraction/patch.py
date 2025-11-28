from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from logging import getLogger
from typing import Any, Optional, Tuple, Union

import torch
from monai.data.utils import iter_patch_position
from pydantic import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    field_validator,
)

from clinicadl.data.structures import DataPoint
from clinicadl.dictionary.words import SAMPLE_POSITION, SAMPLE_TYPE
from clinicadl.utils.config import ObjectConfig

from .base import Extraction, ImplementedExtraction

logger = getLogger("clinicadl.transforms.extraction.patch")


class PadMode(str, Enum):
    "Padding mode for Patch extraction."

    CONSTANT = "constant"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"


class PatchConfig(ObjectConfig["Patch"]):
    """
    Config class for patch extraction.
    """

    patch_size: Tuple[PositiveInt, PositiveInt, PositiveInt]
    overlap: Union[
        Tuple[NonNegativeFloat, NonNegativeFloat, NonNegativeFloat],
        Tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt],
    ]
    pad_mode: Optional[PadMode]
    pad_value: float

    @field_validator("patch_size", "overlap", mode="before")
    @classmethod
    def _ensure_tuple(
        cls,
        value: Any,
    ) -> tuple:
        """
        Ensures that arguments is a tuple.
        """
        if not isinstance(value, Sequence):
            return (value, value, value)
        return value

    @field_validator("overlap", mode="after")
    @classmethod
    def _overlap_validator(cls, value: tuple) -> tuple:
        """Checks that overlap is between 0 and 1 if it is a float."""
        for v in value:
            if isinstance(v, float):
                assert (
                    0 <= v < 1
                ), f"If 'overlap' is a float, it must be between 0 (included) and 1 (excluded). Got {v}"
        return value

    @classmethod
    def _get_class(cls) -> type[Patch]:
        """Returns the class associated to this config class."""
        return Patch


class Patch(Extraction[ObjectConfig]):
    """
    Transform class to extract patches from an image.

    The image is divided into smaller patches using a sliding window approach.

    Adds the following keys to the input :py:class:`~clinicadl.data.structures.DataPoint`:

    - ``patch_location``: tuple[int, int, int]
        The position of the patch in the image, which is defined as the position of its upper left voxel.
        The origin is defined at the upper left voxel of the image.

    Parameters
    ----------
    patch_size : Union[int, Tuple[int, int, int]]
        The size of the patches. If a single value is passed, the same patch size will be used for the three
        spatial dimensions.
    overlap: Union[float, Tuple[float, float, NonNegativeFloat], float, Tuple[int, int, int]], default=0.0
        The amount of overlap between patches. It can be either a ``float`` in :math:`[0.0, 1.0)` that defines relative overlap, or a non-negative ``int`` that defines the
        number of pixels overlapping. If a single value is passed, the same overlap will be used for the three spatial dimensions.
    pad_mode : Optional[PadMode], default="constant"
        A padding mode accepted by :py:func:`torch.nn.functional.pad`, i.e. one of ``"constant"``, ``"reflect"``, ``"replicate"`` or ``"circular"``.
        If ``None``, no padding will be applied, so the patches that cross the border of the image will be dropped.
    pad_value : float, default=0.0
        The value for ``"constant"`` padding.
    """

    config: PatchConfig
    _config_type = PatchConfig

    def __init__(
        self,
        *,
        patch_size: Union[int, Tuple[int, int, int]],
        overlap: Union[
            float,
            Tuple[float, float, float],
            int,
            Tuple[int, int, int],
        ] = 0.0,
        pad_mode: Optional[PadMode] = PadMode.CONSTANT,
        pad_value: float = 0.0,
    ) -> None:
        self.config = PatchConfig(
            patch_size=patch_size,
            overlap=overlap,
            pad_mode=pad_mode,
            pad_value=pad_value,
        )

    @property
    def sample_type(self) -> str:
        """
        The type of the sample returned by this extraction, among {"image", "slice", "patch"}.
        """
        return ImplementedExtraction.PATCH.value.lower()

    def _extract_tensor_sample(
        self, image_tensor: torch.Tensor, sample_position: tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Extracts a single patch from an image.

        Adapted from https://monai-dev.readthedocs.io/en/stable/inferers.html#monai.inferers.SlidingWindowSplitter.__call__.
        """
        spatial_shape = image_tensor.shape[1:]
        pad_size = self._calculate_pad_size(spatial_shape)

        # padding
        if self.config.pad_mode and any(pad_size):
            image_tensor = torch.nn.functional.pad(
                image_tensor,
                pad_size,
                mode=self.config.pad_mode,
                value=self.config.pad_value,
            )

        patch = self._get_patch(
            image_tensor, location=sample_position, patch_size=self.config.patch_size
        )

        return patch

    def _get_sample_positions(
        self, data_point: DataPoint
    ) -> list[tuple[int, int, int]]:
        """
        Returns the positions of the patches in the image.
        """
        spatial_shape = data_point.image.tensor.shape[1:]
        padded_shape = self._get_padded_shape(spatial_shape)
        return list(
            iter_patch_position(
                image_size=padded_shape,
                patch_size=self.config.patch_size,
                overlap=self.config.overlap,
                padded=False,
            )
        )

    def _add_info(
        self, data_point: DataPoint, sample_position: tuple[int, int, int]
    ) -> None:
        """
        Adds relevant info in the datapoint.
        """
        data_point[SAMPLE_TYPE] = self.sample_type
        data_point[SAMPLE_POSITION] = sample_position

    @staticmethod
    def _get_patch(
        tensor: torch.Tensor,
        patch_size: tuple[int, int, int],
        location: tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Gets a patch from a 4D tensor.
        """
        slices = (slice(None),) + tuple(
            slice(loc, loc + ps) for loc, ps in zip(location, patch_size)
        )
        return tensor[slices]

    def _get_padded_shape(
        self, spatial_shape: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        """
        Returns the padded shape from the original shape.
        """
        if not self.config.pad_mode:
            return spatial_shape

        pad_size = self._calculate_pad_size(spatial_shape)
        padded_spatial_shape = tuple(
            shape + pad for shape, pad in zip(spatial_shape, pad_size[1::2])
        )

        return padded_spatial_shape

    def _calculate_pad_size(
        self, spatial_shape: tuple[int, int, int]
    ) -> tuple[int, int, int, int, int, int]:
        """
        Returns the pad size for each dimension.
        """
        pad_size = [0] * 2 * len(spatial_shape)

        if not self.config.pad_mode:
            return pad_size

        for i, sh, ps, ov in zip(
            range(1, len(pad_size), 2),
            spatial_shape,
            self.config.patch_size,
            self.config.overlap,
        ):
            if isinstance(ov, float):
                pad_size[i] = (ps - sh) % round(ps - (ps * ov))
            else:
                pad_size[i] = (ps - sh) % round(ps - ov)

        return pad_size
