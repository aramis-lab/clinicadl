from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, Any, TypeVar

import torch

from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.dictionary.words import SAMPLE_POSITION, SAMPLE_TYPE

from .base import Extraction, ImplementedExtraction

if TYPE_CHECKING:
    from clinicadl.data.structures import DataPoint


logger = getLogger(__name__)

DataPointT = TypeVar("DataPointT", bound="DataPoint")


class ImageConfig(ObjectConfig["Image"]):
    """
    Config class for Image extraction.
    """

    @classmethod
    def _get_class(cls) -> type[Image]:
        """Returns the class associated to this config class."""
        return Image


class Image(Extraction[ImageConfig]):
    """
    Transform class for full image extraction, which is equivalent to
    no extraction.

    Adds the following keys to the input :py:class:`~clinicadl.data.structures.DataPoint`:

    - ``sample_type`` : ``"image"``
    - ``sample_position``: ``None``
        Not relevant here. Only for consistency.
    """

    _config_type = ImageConfig

    def __init__(self):
        self.config = ImageConfig()

    @property
    def sample_type(self) -> str:
        """
        The type of the sample returned by this extraction, among {"image", "slice", "patch"}.
        """
        return ImplementedExtraction.IMAGE.value.lower()

    def _extract_datapoint_from_position(
        self, data_point: DataPointT, sample_position: Any
    ) -> DataPointT:
        self._add_info(data_point, sample_position)

        return data_point

    def _extract_tensor_sample(
        self,
        image_tensor: torch.Tensor,
        sample_position: int,
    ) -> torch.Tensor:
        """
        Returns the entire image tensor as no extraction is performed.
        """
        return image_tensor

    def _get_sample_positions(self, data_point: DataPoint) -> list[int]:
        """
        Returns the positions of the samples in the image, which
        is always [0] here.
        """
        return [0]

    def _add_info(self, data_point: DataPoint, sample_position: int) -> None:
        """
        Adds relevant info in the datapoint.
        """
        data_point[SAMPLE_TYPE] = self.sample_type
        data_point[SAMPLE_POSITION] = None
