from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from logging import getLogger
from typing import TYPE_CHECKING, Any, Generator, Optional, TypeVar, Union, overload

import torch
import torchio as tio

from clinicadl.utils.objects import Config, HasConfig

if TYPE_CHECKING:
    from clinicadl.data.structures import DataPoint

DataPointT = TypeVar("DataPointT", bound="DataPoint")

logger = getLogger("clinicadl.transforms.extraction.base")


class ImplementedExtraction(str, Enum):
    """Extraction methods supported in ClinicaDL."""

    IMAGE = "Image"
    SLICE = "Slice"
    PATCH = "Patch"


class Extraction(HasConfig[Config], ABC):
    """
    Abstract base class for sample extractions.
    """

    @property
    @abstractmethod
    def sample_type(self) -> str:
        """
        The type of the sample returned by this extraction, among {"image", "slice", "patch"}.
        """

    @overload
    def __call__(self, data_point: DataPointT, sample_index: int) -> DataPointT:
        ...

    @overload
    def __call__(
        self, data_point: DataPointT, sample_index: None
    ) -> Generator[DataPointT, None, None]:
        ...

    def __call__(
        self, data_point: DataPointT, sample_index: Optional[int] = None
    ) -> Union[DataPointT, Generator[DataPointT, None, None]]:
        """
        Extracts samples from a :py:class:`~clinicadl.data.structures.DataPoint` object and returns a generator,
        or extracts a single sample and returns a ``DataPoint``.

        Samples are extracted from every images and masks in the input ``DataPoint``.

        Parameters
        ----------
        data_point : DataPoint
            The :py:class:`~clinicadl.data.structures.DataPoint` to perform extraction on.
        sample_index : Optional[int], default=None
            Potential index indicating the sample to extract. If ``None``, a generator of
            all the samples will be returned.

        Returns
        -------
        Union[DataPoint, Generator[DataPoint, None, None]]
            A new ``DataPoint``, with the extracted sample, and some new information about the extraction
            (e.g. the sample position), or a generator of such ``DataPoints``.

        Raises
        ------
        IndexError
            If ``sample_index`` is greater or equal to the number of samples in the image.
        """
        if sample_index is not None:
            return self._extract_datapoint_from_index(data_point, sample_index)
        return self._samples_generator(data_point)

    def num_samples_per_image(self, data_point: DataPoint) -> int:
        """
        Returns the number of samples that can be extracted from an image.

        Parameters
        ----------
        data_point : DataPoint
            The :py:class:`~clinicadl.data.structures.DataPoint` containing the image to perform extraction on.

        Returns
        -------
        int
            The number of samples in the image.
        """
        return len(self._get_sample_positions(data_point))

    def _samples_generator(
        self, data_point: DataPointT
    ) -> Generator[DataPointT, None, None]:
        """
        Returns all the samples via a generator.
        """
        for sample_position in self._get_sample_positions(data_point):
            yield self._extract_datapoint_from_position(data_point, sample_position)

    def _extract_datapoint_from_index(
        self, data_point: DataPointT, sample_index: int
    ) -> DataPointT:
        """
        Extracts a sample from a datapoint (i.e. performs extraction on all
        the images and masks of the DataPoint), given the index of the sample.

        Raises
        ------
        IndexError
            If ``sample_index`` is greater or equal to the number of samples in the image.
        """
        sample_positions = self._get_sample_positions(data_point)
        if sample_index >= len(sample_positions):
            raise IndexError(
                f"'sample_index' {sample_index} is out of range as there are only "
                f"{len(sample_positions)} samples in the image."
            )
        sample_position = sample_positions[sample_index]

        return self._extract_datapoint_from_position(data_point, sample_position)

    def _extract_datapoint_from_position(
        self, data_point: DataPointT, sample_position: Any
    ) -> DataPointT:
        """
        Extracts a sample from a datapoint (i.e. performs extraction on all
        the images and masks of the DataPoint), given the position of the sample.
        """
        samples = deepcopy(data_point)

        image: tio.Image
        for name, image in samples.get_images_dict(intensity_only=False).items():
            try:
                sample = self._extract_tensor_sample(image.tensor, sample_position)
            except IndexError as exc:
                raise IndexError(
                    f"An error occurred while extracting sample located at '{sample_position}' from image '{name}' of ({data_point.participant}, {data_point.session})."
                ) from exc

            image.set_data(sample)

        self._add_info(samples, sample_position)
        samples.applied_transforms = data_point.applied_transforms

        return samples

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

    @abstractmethod
    def _add_info(self, data_point: DataPoint, sample_position: Any) -> None:
        """
        Abstract class to add relevant info in the datapoint.
        """
