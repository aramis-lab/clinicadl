from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from logging import getLogger
from typing import Any, Optional

import torch
import torchio as tio
from pydantic import computed_field

from clinicadl.data.datatypes.preprocessing import Preprocessing
from clinicadl.data.structures import DataPoint
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.typing import PathType

logger = getLogger("clinicadl.transforms.extraction.base")


class ExtractionMethod(str, Enum):
    """Possible extraction methods."""

    IMAGE = "image"
    SLICE = "slice"
    PATCH = "patch"


class Sample(DataPoint):
    """Abstract class for outputs of CapsDataset."""

    image_path: PathType
    extraction: str
    preprocessing: Preprocessing

    @property
    @abstractmethod
    def sample_position(self) -> Optional[int]:
        """The position of the sample."""


class Extraction(ClinicaDLConfig, ABC):
    """
    Abstract base class for sample extractions.
    """

    @computed_field
    @property
    @abstractmethod
    def extract_method(self) -> str:
        """The method to be used for the extraction process (Image, Patch, Slice)."""

    @abstractmethod
    def extract_sample(self, data_point: DataPoint, sample_index: int) -> Sample:
        """
        Abstract main public method to extract a sample from a DataPoint object.

        Parameters
        ----------
        data_point : DataPoint
            The DataPoint to perform extraction on.
        sample_index : int
            Index indicating the sample to extract.

        Returns
        -------
        Sample
            A new DataPoint with the extracted sample for each image
            present in the original DataPoint. The sample extracted from an
            image is accessible via the same name as was the image in the original
            DataPoint.
            The DataPoint also contains some new information about the extraction
            (e.g. the sample index).

        Raises
        ------
        IndexError
            If ``sample_index`` is greater or equal to the number of samples in the images.
        """

    def num_samples_per_image(self, data_point: DataPoint) -> int:
        """
        Returns the number of samples that can be extracted from the input image tensor.

        Parameters
        ----------
        data_point : DataPoint
            The DataPoint containing the image to perform extraction on.

        Returns
        -------
        int
            The number of samples in the image.
        """
        return len(self._get_sample_positions(data_point))

    def _extract_datapoint_sample(
        self, data_point: DataPoint, sample_index: int
    ) -> tuple[DataPoint, Any]:
        """
        Extracts a sample from a DataPoint object, i.e. performs extraction on all
        the images and masks of the DataPoint. Also returns the position of the sample
        in the image.

        Raises
        ------
        IndexError
            If ``sample_index`` is greater or equal to the number of samples in the image.
        """
        extracted_data_point = deepcopy(data_point)

        sample_positions = self._get_sample_positions(data_point)
        if sample_index >= len(sample_positions):
            raise IndexError(
                f"'sample_index' {sample_index} is out of range as there are only "
                f"{len(sample_positions)} samples in the image."
            )
        sample_position = sample_positions[sample_index]

        image: tio.Image
        for name, image in extracted_data_point.get_images_dict(
            intensity_only=False
        ).items():
            try:
                sample = self._extract_tensor_sample(image.tensor, sample_position)
            except IndexError as exc:
                raise IndexError(
                    f"An error occurred while extracting sample '{sample_index}' from image '{name}' of ({data_point.participant}, {data_point.session})."
                ) from exc

            image.set_data(sample)

        return extracted_data_point, sample_position

    @abstractmethod
    def _extract_tensor_sample(
        self,
        image_tensor: torch.Tensor,
        sample_position: Any,
    ) -> torch.Tensor:
        """
        Abstract method for extracting a sample from a given tensor image.
        """

    @abstractmethod
    def _get_sample_positions(
        self,
        data_point: DataPoint,
    ) -> list[Any]:
        """
        Abstract method to get the positions of the samples in the image.
        """
