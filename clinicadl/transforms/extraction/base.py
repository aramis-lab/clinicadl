from abc import abstractmethod
from logging import getLogger
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import torch
from pydantic import PositiveInt

from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.enum import ExtractionMethod

logger = getLogger("clinicadl.base_extraction")

NII_GZ = ".nii.gz"
PT = ".pt"


class BaseExtraction(ClinicaDLConfig):
    """
    Abstract base class for image extraction procedures.

    This class defines the common structure and methods for extracting data from
    neuroimaging files (such as NIfTI) into a tensor representation for further processing.

    Parameters
    ----------
    extract_method : ExtractionMethod
        The method to be used for the extraction process (ROI, Image, Patch, Slice).
    use_uncropped_image : bool, optional
        A flag to specify whether to use the uncropped image, by default True.
    """

    extract_method: ExtractionMethod
    use_uncropped_image: bool = True

    def extract_image(self, input_img: Path) -> torch.Tensor:
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
        try:
            image_array = nib.load(input_img).get_fdata(dtype="float32")  # type: ignore
        except Exception as e:
            raise FileNotFoundError(f"Failed to load the image: {input_img}") from e
        return torch.from_numpy(image_array).unsqueeze(0).float()

    @abstractmethod
    def extract_tensor(
        self,
        image_tensor: torch.Tensor,
        index: int,
    ) -> torch.Tensor:
        """
        Abstract method for extracting specific data from a given image tensor.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The image tensor to extract data from.
        index : int
            Index indicating the element to extract.

        Returns
        -------
        torch.Tensor
            A tensor containing the extracted data.

        Notes
        -----
        This method needs to be implemented in the subclasses.
        """
        pass

    @abstractmethod
    def extract_path(self, image_path, index):
        """
        Abstract method for defining the path where extracted elements will be saved.

        Parameters
        ----------
        image_path : Path
            Path to the original image.
        index : int
            Index of the element being extracted.

        Returns
        -------
        Path
            Path where the extracted data will be saved.

        Notes
        -----
        This method needs to be implemented in the subclasses.
        """
        pass

    @abstractmethod
    def extract(self, nii_path: Path) -> List[Tuple[Path, torch.Tensor]]:
        """
        Abstract method for performing the extraction based on the configured method.

        Parameters
        ----------
        nii_path : Path
            Path to the NIfTI file to process.

        Notes
        -----
        This method needs to be implemented in the subclasses.
        """
        pass

    @abstractmethod
    def num_elem_per_image(self, image: torch.Tensor) -> PositiveInt:
        """
        Abstract method to return the number of extracted elements per image.

        Parameters
        ----------
        image : torch.Tensor
            The image tensor from which the number of elements will be determined.

        Returns
        -------
        PositiveInt
            The number of extracted elements from the image.

        Notes
        -----
        This method needs to be implemented in the subclasses.
        """
        pass
