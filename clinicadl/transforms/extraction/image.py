from logging import getLogger
from pathlib import Path
from typing import Tuple

import torch
from pydantic import PositiveInt

from clinicadl.dataset.transforms.extraction.base import BaseExtraction
from clinicadl.utils.enum import ExtractionMethod

logger = getLogger("clinicadl.extraction.image")

NII_GZ = ".nii.gz"
PT = ".pt"


class Image(BaseExtraction):
    """
    Configuration class for full image extraction as a single tensor.

    This class implements the extraction process for a full image, where the entire
    image is loaded and returned as a single tensor. It handles extraction using
    the `ExtractionMethod.IMAGE` and saves the output as a tensor file.

    Attributes
    ----------
    extract_method : ExtractionMethod
        The method used for the extraction. For this class, it's set to IMAGE.
    """

    extract_method: ExtractionMethod = ExtractionMethod.IMAGE

    def extract(self, nii_path: Path) -> list[Tuple[Path, torch.Tensor]]:
        """
        Extracts the full image as a single tensor file and saves it.

        Parameters
        ----------
        nii_path : Path
            The path to the NIfTI image to be processed.

        Returns
        -------
        list of Tuple[Path, torch.Tensor]
            A list containing a tuple with the output file path and the extracted image tensor.

        Notes
        -----
        The image is loaded, converted into a tensor, and saved with the same name as the original image but with a `.pt` extension.
        """
        image_tensor = self.extract_image(nii_path)
        output_file = nii_path.with_suffix("").with_suffix(PT), image_tensor.clone()
        return [output_file]

    def extract_tensor(
        self,
        image_tensor: torch.Tensor,
        index: int,
    ) -> torch.Tensor:
        """
        Returns the entire image tensor as no further extraction is needed.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The image tensor to extract data from.
        index : int
            The index to identify the extracted data (though this is not used in this method).

        Returns
        -------
        torch.Tensor
            The same image tensor as no further extraction is applied.

        Notes
        -----
        This method is a placeholder in this class as the full image is returned without modification.
        """
        return image_tensor

    def extract_path(self, image_path, index):
        """
        Returns the input image path as the path to save the extracted data.

        Parameters
        ----------
        image_path : Path
            The path to the original image.
        index : int
            The index to identify the extracted data (though this is not used in this method).

        Returns
        -------
        Path
            The path where the extracted data will be saved (same as the input image path).

        Notes
        -----
        This method does not alter the path, returning the same path as the input.
        """
        return image_path

    def num_elem_per_image(self, image: torch.Tensor) -> PositiveInt:
        """
        Returns the number of elements per image. Since the entire image is extracted, this method always returns 1.

        Parameters
        ----------
        image : torch.Tensor
            The image tensor to determine the number of extracted elements.

        Returns
        -------
        PositiveInt
            The number of elements per image, which is always 1 for full image extraction.

        Notes
        -----
        This method is specific to the full image extraction, where only one element (the image) is returned.
        """
        return 1
