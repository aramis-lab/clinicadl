from abc import ABC, abstractmethod
from logging import getLogger
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import torch
from pydantic import computed_field

from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.enum import ExtractionMethod

logger = getLogger("clinicadl.base_extraction")


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
        """The method to be used for the extraction process (ROI, Image, Patch, Slice)."""

    def load_image(self, input_img: Path) -> torch.Tensor:
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
