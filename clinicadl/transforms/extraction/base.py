from abc import ABC, abstractmethod
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import nibabel as nib
import torch
import torchio as tio
from pydantic import computed_field

from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.enum import ExtractionMethod

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

    @abstractmethod
    def extract_sample(
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
        tio_sample: tio.Subject,
        participant_id: str,
        session_id: str,
        image_path: Union[str, Path],
        description: Any,
    ) -> Sample:
        """
        Puts all the output information in a Sample object.

        Parameters
        ----------
        tio_sample : tio.Subject
            a TorchIO Subject, with at least a ScalarImage named 'image'
            and an attribute named 'label'.
        participant_id : str
            the subject concerned.
        session_id : str
            the session concerned.
        image_path : Union[str, Path]
            the path of the base image, from which the sample was extracted.
        description : Any
            a description of the sample (e.g. slice position or patch index).

        Returns
        -------
        Sample
            a Sample object with all the relevant information on the sample.

        Raises
        ------
        AttributeError
            if `tio_sample` doesn't contain a TorchIO ScalarImage named 'image' and an attribute
            'label'.
        """

    def extract_tio_sample(
        self, tio_image: tio.Subject, sample_index: int
    ) -> Tuple[tio.Subject, Any]:
        """
        Extracts a sample from a TorchIO Subject.

        Parameters
        ----------
        tio_image : tio.Subject
            The TorchIO Subject to perform extraction on.
        sample_index : int
            Index indicating the sample to extract.

        Returns
        -------
        tio.Subject
            A new TorchIO Subject with the extracted samples for each image
            present in the original `tio_image`. The sample extracted from an
            image is accessible via the same name as was the image in the original
            `tio_image`.
        Any
            A description of the sample (e.g. slice position or patch index).

        Raises
        ------
        ValueError
            If all the images in `tio_image` don't have the same shape.
        IndexError
            If 'sample_index' is greater or equal to the number of samples in the images.
        """
        tio_sample = deepcopy(tio_image)

        image: tio.Image
        for i, (name, image) in enumerate(
            tio_image.get_images_dict(intensity_only=False).items()
        ):
            if i == 0:
                shape = image.tensor.shape
                description = self._get_sample_description(image.tensor, sample_index)
            if image.tensor.shape != shape:  # pylint: disable=possibly-used-before-assignment
                raise ValueError(
                    f"Got images of different size in 'tio_image': {tio_image.get_images_names()[0]} is "
                    f"{shape}, whereas {name} is {image.tensor.shape}"
                )

            sample = self.extract_sample(image.tensor, sample_index)

            if isinstance(image, tio.ScalarImage):
                tio_sample.add_image(tio.ScalarImage(tensor=sample), name)
            elif isinstance(image, tio.LabelMap):
                tio_sample.add_image(tio.LabelMap(tensor=sample), name)

        return tio_sample, description  # pylint: disable=possibly-used-before-assignment

    @staticmethod
    def _check_tio_subject(tio_subject: tio.Subject):
        """
        Checks that a TorchIO Subject is valid, i.e. a Subject with a TorchIO ScalarImage
        named 'image' and a label named 'label'.
        """
        if not hasattr(tio_subject, "image") or not isinstance(
            tio_subject.image, tio.ScalarImage
        ):
            raise AttributeError(
                "The TorchIO Subject must contain a ScalarImage named 'image'. Got only the following images: "
                f"{tio_subject.get_images_names()}"
            )
        if not hasattr(tio_subject, "label"):
            raise AttributeError(
                "The TorchIO Subject must contain an attribute named 'label'."
            )
