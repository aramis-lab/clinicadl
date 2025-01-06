from abc import ABC, abstractmethod
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
import torchio as tio
from pydantic import computed_field

from clinicadl.data.structures import DataPoint
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.enum import ExtractionMethod
from clinicadl.utils.typing import PathType

logger = getLogger("clinicadl.base_extraction")


class Sample(ClinicaDLConfig, ABC):
    """Abstract class for outputs of CapsDataset."""

    sample: torch.Tensor
    participant_id: str
    session_id: str
    image_path: str
    label: Optional[Union[float, int, torch.Tensor]]

    @computed_field
    @property
    @abstractmethod
    def extraction(self) -> ExtractionMethod:
        """The extraction method."""


class Extraction(ClinicaDLConfig, ABC):
    """
    Abstract base class for image extraction procedures.

    This class defines the common structure and methods for extracting data from
    neuroimaging files (such as NIfTI) into a tensor representation for further processing.
    """

    @computed_field
    @property
    @abstractmethod
    def extract_method(self) -> ExtractionMethod:
        """The method to be used for the extraction process (Image, Patch, Slice)."""

    # TODO : put in private?
    @abstractmethod
    def extract_tensor_sample(
        self,
        image_tensor: torch.Tensor,
        sample_index: int,
    ) -> torch.Tensor:
        """
        Abstract method for extracting a sample from a given image.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The image tensor to extract a sample from.
        sample_index : int
            Index indicating the sample to extract.

        Returns
        -------
        torch.Tensor
            A tensor containing the extracted sample.

        Raises
        ------
        IndexError
            If 'sample_index' is greater or equal to the number of samples in the image.

        Notes
        -----
        This method needs to be implemented in the subclasses.
        """

    # TODO : remove?
    @abstractmethod
    def sample_path(self, image_path: Path, sample_index: int) -> Path:
        """
        Abstract method for defining the path where extracted sample will be saved.

        Parameters
        ----------
        image_path : Path
            Path to the original image.
        sample_index : int
            Index of the sample being extracted.

        Returns
        -------
        Path
            Path where the extracted sample will be saved.

        Notes
        -----
        This method needs to be implemented in the subclasses.
        """

    # TODO : remove?
    @abstractmethod
    def extract(self, nii_path: Path) -> List[Tuple[Path, torch.Tensor]]:
        """
        Abstract method to extract all the samples.

        Parameters
        ----------
        nii_path : Path
            Path to the NIfTI file to process.

        Returns
        -------
        List[Tuple[Path, torch.Tensor]]
            A list of tuples, where each tuple contains an extracted sample,
            and the path where to store it.

        Notes
        -----
        This method needs to be implemented in the subclasses.
        """

    @abstractmethod
    def num_samples_per_image(self, image: torch.Tensor) -> int:
        """
        Abstract method to return the number of extracted samples per image.

        Parameters
        ----------
        image : torch.Tensor
            The image tensor from which the number of samples will be determined.

        Returns
        -------
        int
            The number of samples in the image.

        Notes
        -----
        This method needs to be implemented in the subclasses.
        """

    @abstractmethod
    def _get_sample_description(
        self, image_tensor: torch.Tensor, sample_index: int
    ) -> Any:
        """A description of the sample (e.g. slice position or patch index)."""

    @abstractmethod
    def format_output(
        self,
        data_point: DataPoint,
        participant_id: str,
        session_id: str,
        image_path: PathType,
        description: Any,
    ) -> Sample:
        """
        Puts all the output information in a Sample object.

        Parameters
        ----------
        data_point : DataPoint
            the `DataPoint` object associated to the sample.
        participant_id : str
            the subject concerned.
        session_id : str
            the session concerned.
        image_path : PathType
            the path of the base image, from which the sample was extracted.
        description : Any
            a description of the sample (e.g. slice position or patch index).

        Returns
        -------
        Sample
            a Sample object with all the relevant information on the sample.
        """

    def extract_sample(
        self, data_point: DataPoint, sample_index: int
    ) -> Tuple[DataPoint, Any]:
        """
        Extracts a sample from a DataPoint object.

        Parameters
        ----------
        data_point : DataPoint
            The DataPoint to perform extraction on.
        sample_index : int
            Index indicating the sample to extract.

        Returns
        -------
        DataPoint
            A new DataPoint with the extracted samples for each image
            present in the original `data_point`. The sample extracted from an
            image is accessible via the same name as was the image in the original
            `data_point`.
        Any
            A description of the sample (e.g. slice position or patch index).

        Raises
        ------
        IndexError
            If 'sample_index' is greater or equal to the number of samples in the images.
        """
        extracted_data_point = deepcopy(data_point)

        image: tio.Image
        for name, image in data_point.get_images_dict(intensity_only=False).items():
            sample = self.extract_tensor_sample(image.tensor, sample_index)

            if isinstance(image, tio.ScalarImage):
                extracted_data_point.add_image(tio.ScalarImage(tensor=sample), name)
            elif isinstance(image, tio.LabelMap):
                extracted_data_point.add_image(tio.LabelMap(tensor=sample), name)

        description = self._get_sample_description(
            data_point.image.tensor, sample_index
        )

        return extracted_data_point, description
