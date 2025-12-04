from __future__ import annotations

from logging import getLogger

import torch

from clinicadl.data.structures import DataPoint
from clinicadl.dictionary.words import SAMPLE_POSITION, SAMPLE_TYPE
from clinicadl.utils.config import ObjectConfig

from .base import Extraction, ImplementedExtraction

logger = getLogger("clinicadl.transforms.extraction.image")


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
