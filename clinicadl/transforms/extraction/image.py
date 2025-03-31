from logging import getLogger
from typing import Any, Tuple

import torch
import torchio as tio
from pydantic import computed_field

from clinicadl.data.structures import DataPoint
from clinicadl.utils.enum import ExtractionMethod
from clinicadl.utils.typing import PathType

from .base import Extraction, Sample

logger = getLogger("clinicadl.transforms.extraction.image")


class ImageSample(Sample):
    """
    Output of a CapsDataset when no extraction is performed and
    thus the full image is returned.

    Attributes
    ----------
    sample : torch.Tensor
        The image as a 4D PyTorch tensor (with one channel dimension).
    affine : np.ndarray
        The affine matrix associated to the image.
    participant : str
        The participant concerned.
    session : str
        The session concerned.
    image_path : str
        The path to the image.
    label : Optional[Union[float, int, torch.Tensor]]
        The potential label associated to the image.
    """

    @computed_field
    @property
    def extraction(self) -> str:
        """The extraction method."""
        return ExtractionMethod.IMAGE.value

    @computed_field
    @property
    def id(self) -> int:
        """The index of the sample. Always 1 here."""
        return 1


class Image(Extraction):
    """
    Transform class for full image extraction, which is equivalent to
    no extraction.
    """

    @computed_field
    @property
    def extract_method(self) -> str:
        """The method to be used for the extraction process (Image, Patch, Slice)."""
        return ExtractionMethod.IMAGE.value

    def extract_sample(
        self, data_point: DataPoint, sample_index: int = 0
    ) -> Tuple[DataPoint, None]:
        """
        Extracts a sample from a DataPoint. For compatibility,
        as no extraction is performed with Image.

        Parameters
        ----------
        data_point : DataPoint
            The DataPoint to perform extraction on.
        sample_index : int (optional, default=0)
            Index indicating the sample to extract. For compatibility,
            must be left to 0 here.

        Returns
        -------
        DataPoint
            A copy of the input `data_point`, as no extraction is performed.
        None
            Sample description. Always None here, as no description of the
            sample is needed.

        Raises
        ------
        IndexError
            If 'sample_index' is not 0.
        """
        return super().extract_sample(data_point, sample_index)

    def num_samples_per_image(self, image: torch.Tensor) -> int:
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
        """
        return 1

    def format_output(
        self,
        data_point: DataPoint,
        image_path: PathType,
        description: Any = None,
    ) -> ImageSample:
        """
        Puts all the output information in an ImageSample object.

        Parameters
        ----------
        data_point : DataPoint
            the `DataPoint` object associated to the image.
        image_path : PathType
            the path of the image.
        description : Any (optional, default=None)
            a description of the sample. For compatibility, not necessary in the case of Image.

        Returns
        -------
        ImageSample
            an ImageSample object with all the relevant information on the image.
        """
        return ImageSample(
            sample=data_point.image.tensor,
            affine=data_point.image.affine,
            participant=data_point.participant,
            session=data_point.session,
            image_path=str(image_path),
            label=data_point.label.tensor
            if isinstance(data_point.label, tio.LabelMap)
            else data_point.label,
        )

    def _extract_tensor_sample(
        self,
        image_tensor: torch.Tensor,
        sample_index: int = 0,
    ) -> torch.Tensor:
        """
        Returns the entire image tensor as no further extraction is needed.

        Raises
        ------
        IndexError
            If 'sample_index' is not 0.
        """
        if sample_index != 0:
            raise IndexError(
                f"'sample_index' {sample_index} is out of range as there is only "
                "1 sample in the image."
            )

        return image_tensor.clone()

    def _get_sample_description(
        self, image_tensor: torch.Tensor, sample_index: int
    ) -> None:
        """No need for description in the case of image extraction."""
        return None
