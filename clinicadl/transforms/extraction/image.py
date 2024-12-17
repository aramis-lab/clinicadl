from logging import getLogger
from pathlib import Path
from typing import List, Tuple, Union

import torch
import torchio as tio
from pydantic import PositiveInt, computed_field

from clinicadl.dictionary.suffixes import PT
from clinicadl.utils.enum import ExtractionMethod

from .base import Extraction, Sample

logger = getLogger("clinicadl.extraction.image")


class ImageSample(Sample):
    """
    Output of a CapsDataset when image extraction is performed (i.e. no extraction).

    Attributes
    ----------
    sample : torch.Tensor
        the image as 4D PyTorch tensor (with one channel dimension).
    participant_id : str
        the subject concerned.
    session_id : str
        the session concerned.
    image_path : str
        the path to the image.
    label : Optional[Union[float, int, torch.Tensor]]
        the potential label associated to the image.
    """

    @computed_field
    @property
    def extraction(self) -> ExtractionMethod:
        """The extraction method."""
        return ExtractionMethod.IMAGE


class Image(Extraction):
    """
    Transform class for full image extraction as a single tensor.

    This class implements the extraction process to get the full image, where the entire
    image is loaded and returned as a single tensor.
    """

    @computed_field
    @property
    def extract_method(self) -> ExtractionMethod:
        """The method to be used for the extraction process (Image, Patch, Slice)."""
        return ExtractionMethod.IMAGE

    def extract(self, nii_path: Path) -> List[Tuple[Path, torch.Tensor]]:
        """
        Extracts the full image as a single tensor file and returns the path
        where to save it.

        Parameters
        ----------
        nii_path : Path
            The path to the NIfTI image to be processed.

        Returns
        -------
        List[Tuple[Path, torch.Tensor]]
            A list containing a single tuple with the output file path and the extracted image tensor.

        Notes
        -----
        The image is loaded and returned into a tensor along with the input path with the `.pt` extension.
        """
        image_tensor = self.load_image(nii_path)

        return [(self.sample_path(nii_path), self.extract_sample(image_tensor))]

    def extract_sample(
        self,
        image_tensor: torch.Tensor,
        sample_index: int = 0,
    ) -> torch.Tensor:
        """
        Returns the entire image tensor as no further extraction is needed.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The image tensor to extract data from.
        sample_index : int (optional, default=0)
            The index to identify the extracted data (though this is not used in this method).

        Returns
        -------
        torch.Tensor
            The same image tensor as no further extraction is applied.

        Raises
        ------
        IndexError
            If 'sample_index' is not 0.

        Notes
        -----
        This method is a placeholder in this class as the full image is returned without modification.
        """
        if sample_index != 0:
            raise IndexError(
                f"'sample_index' {sample_index} is out of range as there is only "
                "1 sample in the image."
            )

        return image_tensor.clone()

    def sample_path(self, image_path: Path, sample_index: int = 0) -> Path:  # pylint:disable=unused-argument
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
        This method only changes the extension of the path.
        """
        return image_path.with_suffix("").with_suffix(PT)

    def num_samples_per_image(self, image: torch.Tensor) -> PositiveInt:
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

    def _get_sample_description(
        self, image_tensor: torch.Tensor, sample_index: int
    ) -> None:
        """No need for description in the case of image extraction."""
        return None

    def format_output(
        self,
        tio_sample: tio.Subject,
        participant_id: str,
        session_id: str,
        image_path: Union[str, Path],
    ) -> ImageSample:
        """
        Puts all the output information in an ImageSample object.

        Parameters
        ----------
        tio_sample : tio.Subject
            a TorchIO Subject corresponding to the image, with at least a ScalarImage named 'sample',
            an attribute named 'label' and an attribute named 'description'.
        participant_id : str
            the subject concerned.
        session_id : str
            the session concerned.
        image_path : Union[str, Path]
            the path of the image.

        Returns
        -------
        ImageSample
            an ImageSample object with all the relevant information on the image.

        Raises
        ------
        AttributeError
            if `tio_sample` doesn't have a TorchIO ScalarImage named 'sample', and attributes
            'label' and 'description'.
        """
        self._check_tio_sample(tio_sample)

        sample = tio_sample.sample.tensor
        if isinstance(tio_sample.label, tio.Image):
            label = tio_sample.label.tensor
        else:
            label = tio_sample.label

        return ImageSample(
            sample=sample,
            participant_id=participant_id,
            session_id=session_id,
            image_path=str(image_path),
            label=label,
        )

    def extract_tio_sample(
        self, tio_image: tio.Subject, sample_index: int = 0
    ) -> tio.Subject:
        """
        Converts a TorchIO Subject representing the image to a TorchIO Subject
        representing the sample (which is here the image).

        Parameters
        ----------
        tio_image : tio.Subject
            The image as a TorchIO Subject. Can contain masks associated
            to the image as well.
        sample_index : int (optional, default=0)
            For consistency with other extraction methods. Always 0 here.

        Returns
        -------
        tio.Subject
            A new TorchIO Subject representing the sample (the full image here), accessible via the
            attribute 'sample', and the potential masks, extracted in the same way as the sample.

        Raises
        ------
        AttributeError
            If 'tio_image' doesn't have a TorchIO ScalarImage named 'image'.
        IndexError
            If 'sample_index' is not 0.

        Notes
        -----
        This method is trivial here as no extraction is performed. The TorchIO Subject is just converted
        to another format.
        """
        return super().extract_tio_sample(tio_image, sample_index)
