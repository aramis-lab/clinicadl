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
    computed_field,
    field_validator,
)

from clinicadl.data.structures import DataPoint

from .base import Extraction, ExtractionMethod, Sample

logger = getLogger("clinicadl.transforms.extraction.patch")


class PadMode(str, Enum):
    "Padding mode for Patch extraction."

    CONSTANT = "constant"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"


class PatchSample(Sample):
    """
    Output of a CapsDataset when patch extraction is performed (i.e.
    when :py:class:`~Patch` is used).

    It is simply a :py:class:`~clinicadl.data.structures.DataPoint`, with
    additional information on the patch extraction.

    Attributes
    ----------
    image : torchio.ScalarImage
        The patch, as a :py:class:`torchio.ScalarImage`.
    label : Optional[Union[float, int, torchio.LabelMap]]
        The label associated to the patch. Can be a ``float`` (regression),
        an ``int`` (classification), a mask (as a :py:class:`torchio.LabelMap`; for segmentation)
        or ``None`` if no label (reconstruction). If the label is a mask, patch extraction
        was also performed on it.
    participant : str
        The participant concerned.
    session : str
        The session concerned.
    preprocessing : Preprocessing
        The proprocessing of the image (see :ref:`api_data_types`).
    image_path : Union[str, Path]
        The path to the image.
    patch_location : Tuple[int, int, int]
        The position of the patch in the image, which is defined as the position of its upper left voxel.
    """

    patch_location: Tuple[int, int, int]

    @property
    def sample_position(self) -> int:
        """The position of the sample."""
        return self.patch_location


class Patch(Extraction):
    """
    Transform class to extract patches from an image.

    The image is divided into smaller patches using a sliding window approach.

    Adds the following keys to the input :py:class:`~clinicadl.data.structures.DataPoint`:

    - ``patch_location``: tuple[int, int, int]
        The position of the patch in the image, which is defined as the position of its upper left voxel.
        The origin is defined at the upper left voxel of the image.

    Parameters
    ----------
    patch_size : Union[PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]]
        The size of the patches. If a single value is passed, the same patch size will be used for the three
        spatial dimensions.
    overlap: Union[NonNegativeFloat, Tuple[NonNegativeFloat, NonNegativeFloat, NonNegativeFloat], NonNegativeInt, Tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt]]
        The amount of overlap between patches. It can be either a ``float`` in :math:`[0.0, 1.0)` that defines relative overlap, or a non-negative ``int`` that defines the
        number of pixels overlapping. If a single value is passed, the same overlap will be used for the three spatial dimensions.
    pad_mode : Optional[PadMode], default="constant"
        A padding mode accepted by :py:func:`torch.nn.functional.pad`, i.e. one of ``"constant"``, ``"reflect"``, ``"replicate"`` or ``"circular"``.
        If ``None``, no padding will be applied, so the patches that cross the border of the image will be dropped.
    pad_value : float, default=0.0
        The value for ``"constant"`` padding.
    """

    patch_size: Tuple[PositiveInt, PositiveInt, PositiveInt]
    overlap: Union[
        Tuple[NonNegativeFloat, NonNegativeFloat, NonNegativeFloat],
        Tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt],
    ]
    pad_mode: Optional[PadMode]
    pad_value: float

    def __init__(
        self,
        *,
        patch_size: Union[PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]],
        overlap: Union[
            NonNegativeFloat,
            Tuple[NonNegativeFloat, NonNegativeFloat, NonNegativeFloat],
            NonNegativeInt,
            Tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt],
        ] = 0.0,
        pad_mode: Optional[PadMode] = PadMode.CONSTANT,
        pad_value: float = 0.0,
    ) -> None:
        super().__init__(
            patch_size=self._ensure_tuple(patch_size),
            overlap=self._ensure_tuple(overlap),
            pad_mode=pad_mode,
            pad_value=pad_value,
        )

    @computed_field
    @property
    def extract_method(self) -> str:
        """The method to be used for the extraction process (Image, Patch, Slice)."""
        return ExtractionMethod.PATCH.value

    @staticmethod
    def _ensure_tuple(
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

    def extract_sample(self, data_point: DataPoint, sample_index: int) -> PatchSample:
        """
        Extracts a patch from a DataPoint.

        Parameters
        ----------
        data_point : DataPoint
            The DataPoint to perform extraction on.
        sample_index : int
            Index indicating the patch to extract.

        Returns
        -------
        DataPoint
            A new DataPoint object with the extracted patches for each image
            present in the original ``data_point``. The patch extracted from an
            image is accessible via the same name as was the image in the original
            ``data_point``.
            Additional information on the extraction is added.

        Raises
        ------
        IndexError
            If ``sample_index`` is greater or equal to the number of patches in the images.
        """
        extracted_datapoint, sample_position = self._extract_datapoint_sample(
            data_point, sample_index
        )
        sample = PatchSample(
            **extracted_datapoint,
            extraction=self.extract_method,
            patch_location=sample_position,
        )
        sample.applied_transforms = extracted_datapoint.applied_transforms

        return sample

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
                patch_size=self.patch_size,
                overlap=self.overlap,
                padded=False,
            )
        )

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
        if self.pad_mode and any(pad_size):
            image_tensor = torch.nn.functional.pad(
                image_tensor,
                pad_size,
                mode=self.pad_mode,
                value=self.pad_value,
            )

        patch = self._get_patch(
            image_tensor, location=sample_position, patch_size=self.patch_size
        )

        return patch

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
        if not self.pad_mode:
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

        if not self.pad_mode:
            return pad_size

        for i, sh, ps, ov in zip(
            range(1, len(pad_size), 2), spatial_shape, self.patch_size, self.overlap
        ):
            if isinstance(ov, float):
                pad_size[i] = (ps - sh) % round(ps - (ps * ov))
            else:
                pad_size[i] = (ps - sh) % round(ps - ov)

        return pad_size
