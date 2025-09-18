from logging import getLogger

import torch
from pydantic import computed_field

from clinicadl.data.structures import DataPoint

from .base import Extraction, ExtractionMethod, Sample

logger = getLogger("clinicadl.transforms.extraction.image")


class ImageSample(Sample):
    """
    Output of a CapsDataset when no extraction is performed and
    thus the full image is returned.

    Here it is simply a :py:class:`~clinicadl.data.structures.DataPoint`, with an
    additional information on the path of the image (``image_path``).

    Attributes
    ----------
    image : torchio.ScalarImage
        The image, as a :py:class:`torchio.ScalarImage`.
    label : Optional[Union[float, int, torchio.LabelMap]]
        The label associated to the image. Can be a ``float`` (regression),
        an ``int`` (classification), a mask (as a :py:class:`torchio.LabelMap`; for segmentation)
        or ``None`` if no label (reconstruction).
    participant : str
        The participant concerned.
    session : str
        The session concerned.
    preprocessing : Preprocessing
        The proprocessing of the image (see :ref:`api_data_types`).
    image_path : Union[str, Path]
        The path to the image.
    """

    @property
    def _sample_index(self) -> int:
        """The index of the sample. Equal to 1 here."""
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
    ) -> ImageSample:
        """
        Extracts a sample from a DataPoint. For compatibility,
        as no extraction is performed with Image.

        Parameters
        ----------
        data_point : DataPoint
            The DataPoint to perform extraction on.
        sample_index : int, default=0
            Index indicating the sample to extract. For compatibility,
            must be left to 0 here.

        Returns
        -------
        ImageSample
            A copy of the input DataPoint, as no extraction is performed.
            This copy is put in a :py:func:`~ImageSample`.

        Raises
        ------
        IndexError
            If ``sample_index`` is not 0.
        """
        extracted_datapoint = self._extract_datapoint_sample(data_point, sample_index)
        sample = ImageSample(
            **extracted_datapoint,
            extraction=self.extract_method,
        )
        sample.applied_transforms = extracted_datapoint.applied_transforms

        return sample

    def num_samples_per_image(self, data_point: DataPoint) -> int:
        """
        Returns the number of elements per image. Since no extraction is performed, this method always returns 1.

        Parameters
        ----------
        data_point : DataPoint
            The DataPoint containing the image to perform extraction on.

        Returns
        -------
        int
            The number of elements per image, which is always 1 for full image extraction.
        """
        return 1

    def _extract_tensor_sample(
        self,
        image_tensor: torch.Tensor,
        sample_position: int,
    ) -> torch.Tensor:
        """
        Returns the entire image tensor as no extraction is performed.
        """
        return image_tensor

    def _get_sample_position(
        self,
        data_point: DataPoint,
        sample_index: int,
    ) -> int:
        """
        To get the position of the sample in the image, which is always 0 here.

        Raises
        ------
        IndexError
            If ``sample_index`` is not 0.
        """
        if sample_index != 0:
            raise IndexError(
                f"sample_index '{sample_index}' is out of range as there is only "
                "1 sample in the image."
            )

        return 0
