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

    @staticmethod
    def load_image(input_img: Path) -> torch.Tensor:
        """
        Loads a NIfTI image and converts it to a float32 tensor.

        Parameters
        ----------
        input_img : Path
            The path to the input NIfTI image.

        Returns
        -------
        torch.Tensor
            A tensor representing the image with shape [1, C, H, W].

        Raises
        ------
        FileNotFoundError
            If the provided image path does not exist or cannot be read.
        nib.loadsave.ImageFileError
            If the image file cannot be read as a NIfTI file.
        """
        if not Path(input_img).exists():
            raise FileNotFoundError(f"The path '{input_img}' does not match any file.")

        try:
            image_array = nib.load(input_img).get_fdata(dtype="float32")  # type: ignore
        except Exception as e:
            raise Exception(
                f"Unable to read the image in {input_img}. Consider using a nifti file format "
                "('.nii' or '.nii.gz')."
            ) from e

        return torch.from_numpy(image_array).unsqueeze(0).float()

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
        """A description of the sample, e.g. slice position or patch index."""

    @abstractmethod
    def format_output(
        self,
        tio_sample: tio.Subject,
        participant_id: str,
        session_id: str,
        image_path: Union[str, Path],
    ) -> Sample:
        """
        Puts all the output information in a Sample object.

        Parameters
        ----------
        tio_sample : tio.Subject
            a TorchIO Subject corresponding to the sample, with at least a ScalarImage named 'sample',
            an attribute named 'label' and an attribute named 'description'.
        participant_id : str
            the subject concerned.
        session_id : str
            the session concerned.
        image_path : Union[str, Path]
            the path of the base image, from which the sample was extracted.

        Returns
        -------
        Sample
            a Sample object with all the relevant information on the sample.

        Raises
        ------
        AttributeError
            if `tio_sample` doesn't have a TorchIO ScalarImage named 'sample', and attributes
            'label' and 'description'.
        """

    def extract_tio_sample(
        self, tio_image: tio.Subject, sample_index: int
    ) -> tio.Subject:
        """
        Extracts a sample from a TorchIO Subject.

        Parameters
        ----------
        tio_image : tio.Subject
            The image as a TorchIO Subject. Can contain masks associated
            to the image as well.
        sample_index : int
            Index indicating the sample to extract.

        Returns
        -------
        tio.Subject
            A new TorchIO Subject with the extracted sample, accessible via the attribute
            'sample', and the potential masks, extracted in the same way as the sample.

        Raises
        ------
        AttributeError
            If 'tio_image' doesn't have a TorchIO ScalarImage named 'image'.
        IndexError
            If 'sample_index' is greater or equal to the number of samples in the image.
        """
        if not hasattr(tio_image, "image") or not isinstance(
            tio_image.image, tio.ScalarImage
        ):
            raise AttributeError(
                "'tio_image' must contain ScalarImage named 'image'. Got only the following images: "
                f"{tio_image.get_images_names()}"
            )

        tio_sample = deepcopy(tio_image)

        image: tio.Image
        for name, image in tio_image.get_images_dict(intensity_only=False).items():
            sample = self.extract_sample(image.tensor, sample_index)

            if isinstance(image, tio.ScalarImage):
                setattr(tio_sample, name, tio.ScalarImage(tensor=sample))
            elif isinstance(image, tio.LabelMap):
                setattr(tio_sample, name, tio.LabelMap(tensor=sample))

            tio_sample.description = self._get_sample_description(
                image.tensor, sample_index
            )

        tio_sample.sample = tio_sample.image
        delattr(tio_sample, "image")

        return tio_sample

    @staticmethod
    def _check_tio_sample(tio_sample: tio.Subject):
        """
        Checks that a TorchIO Subject is a valid sample, i.e. a sample with a TorchIO ScalarImage
        named 'sample', a label named 'label' and a description named 'description'.
        """
        if not hasattr(tio_sample, "sample") or not isinstance(
            tio_sample.sample, tio.ScalarImage
        ):
            raise AttributeError(
                "'tio_sample' must contain ScalarImage named 'image'. Got only the following images: "
                f"{tio_sample.get_images_names()}"
            )
        if not hasattr(tio_sample, "label"):
            raise AttributeError(
                "'tio_sample' must contain an attribute named 'label'."
            )
        if not hasattr(tio_sample, "description"):
            raise AttributeError(
                "'tio_sample' must contain an attribute named 'description'."
            )
