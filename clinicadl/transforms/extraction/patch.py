from logging import getLogger
from typing import Tuple, Union

import torch
from pydantic import PositiveInt, computed_field

from clinicadl.data.structures import DataPoint

from .base import Extraction, ExtractionMethod, Sample

logger = getLogger("clinicadl.transforms.extraction.patch")


class PatchSample(Sample):
    """
    Output of a CapsDataset when patch extraction is performed (i.e.
    when :py:class:`~Patch` is used).

    It is simply a :py:class:`~clinicadl.data.structures.DataPoint`, with
    additional information on the patch extraction.

    Attributes
    ----------
    image : torchio.ScalarImage
        The patch, as a :py:class:`torchio.ScalarImage`.
    label : Optional[Union[float, int, torchio.LabelMap]]
        The label associated to the patch. Can be a ``float`` (regression),
        an ``int`` (classification), a mask (as a :py:class:`torchio.LabelMap`; for segmentation)
        or ``None`` if no label (reconstruction). If the label is a mask, patch extraction
        was also performed on it.
    participant : str
        The participant concerned.
    session : str
        The session concerned.
    preprocessing : Preprocessing
        The proprocessing of the image (see :ref:`api_data_types`).
    image_path : Union[str, Path]
        The path to the image.
    patch_index : int
        The index of the patch among all patches extracted from the image.
    patch_size : Tuple[int, int, int]
        The size of the patch.
    patch_stride : Tuple[int, int, int]
        The stride used for patch extraction.
    """

    patch_index: int
    patch_size: Tuple[int, int, int]
    patch_stride: Tuple[int, int, int]

    @property
    def _sample_index(self) -> int:
        """The index of the sample. Equal to 'patch_index' here."""
        return self.patch_index


class Patch(Extraction):
    """
    Transform class to extract patches from an image.

    The image is divided into smaller patches using a sliding window approach, where the patch size
    and the stride are configurable.

    Adds the following keys to the input :py:class:`~clinicadl.data.structures.DataPoint`:

    - ``patch_index``: int
        The index of the patch among all patches extracted from the image.
    - ``patch_size``: Tuple[int, int, int]
        The size of the patch.
    - ``patch_stride``: Tuple[int, int, int]
        The stride used for patch extraction.

    Parameters
    ----------
    patch_size :  Union[PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]], default=50
        The size of each patch. If a single value is passed, the same patch size will be used for the three
        spatial dimensions.
    stride : Union[PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]], default=50
        The stride or step size used to move the sliding window. If a single value is passed, the same patch
        stride will be used for the three spatial dimensions.
    """

    patch_size: Tuple[PositiveInt, PositiveInt, PositiveInt]
    stride: Tuple[PositiveInt, PositiveInt, PositiveInt]

    def __init__(
        self,
        *,
        patch_size: Union[
            PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]
        ] = 50,
        stride: Union[PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]] = 50,
    ) -> None:
        super().__init__(
            patch_size=self._ensure_tuples(patch_size),
            stride=self._ensure_tuples(stride),
        )

    @computed_field
    @property
    def extract_method(self) -> str:
        """The method to be used for the extraction process (Image, Patch, Slice)."""
        return ExtractionMethod.PATCH.value

    @staticmethod
    def _ensure_tuples(
        value: Union[PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]],
    ) -> Tuple[PositiveInt, PositiveInt, PositiveInt]:
        """
        Ensures that 'patch_size' and 'stride' are always tuples.
        """
        if isinstance(value, int):
            return (value, value, value)
        else:
            return value

    def extract_sample(self, data_point: DataPoint, sample_index: int) -> PatchSample:
        """
        Extracts a patch from a DataPoint.

        Parameters
        ----------
        data_point : DataPoint
            The DataPoint to perform extraction on.
        sample_index : int
            Index indicating the patch to extract.

        Returns
        -------
        DataPoint
            A new DataPoint object with the extracted patches for each image
            present in the original ``data_point``. The patch extracted from an
            image is accessible via the same name as was the image in the original
            ``data_point``.
            Additional information on the extraction is added.

        Raises
        ------
        IndexError
            If ``sample_index`` is greater or equal to the number of patches in the images.
        """
        extracted_datapoint = self._extract_datapoint_sample(data_point, sample_index)
        sample = PatchSample(
            **extracted_datapoint,
            extraction=self.extract_method,
            patch_index=sample_index,
            patch_size=self.patch_size,
            patch_stride=self.stride,
        )
        sample.applied_transforms = extracted_datapoint.applied_transforms

        return sample

    def num_samples_per_image(self, data_point: DataPoint) -> int:
        """
        Returns the total number of patches extracted from an image.

        Parameters
        ----------
        data_point : DataPoint
            The DataPoint containing the image to perform extraction on.

        Returns
        -------
        int
            The total number of patches that can be extracted from the image.

        Notes
        -----
        The number of patches is determined by the image size, the patch size, and the stride.
        """
        return self._get_patches(data_point.image.tensor).shape[1]

    def _extract_tensor_sample(
        self, image_tensor: torch.Tensor, sample_position: int
    ) -> torch.Tensor:
        """
        Extracts a single patch from an image.
        """
        patches_tensor = self._get_patches(image_tensor)

        return patches_tensor[:, sample_position]

    def _get_patches(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Creates a tensor of patches from the image using the PyTorch method :py:meth:`torch.Tensor.unfold`.

        Returns
        -------
        torch.Tensor
            A tensor containing all the patches extracted from the image. The tensor shape
            will be ``(num_patches, patch_size[0], patch_size[1], patch_size[2])``, where ``num_patches`` is
            determined by the image size, the patch size, and the stride.
        """
        n_channels, *spatial_shape = image_tensor.shape
        if self.patch_size > tuple(spatial_shape):
            raise IndexError(
                "The patch size can't be greater than the size of the image. "
                f"Got image with spatial shape {tuple(spatial_shape)}, but patch size is {self.patch_size}."
            )

        patches_tensor = (
            image_tensor.unfold(1, self.patch_size[0], self.stride[0])
            .unfold(2, self.patch_size[1], self.stride[1])
            .unfold(3, self.patch_size[2], self.stride[2])
            .contiguous()
        )

        return patches_tensor.view(
            n_channels, -1, self.patch_size[0], self.patch_size[1], self.patch_size[2]
        )

    def _get_sample_position(
        self,
        data_point: DataPoint,
        sample_index: int,
    ) -> int:
        """
        To get the position of the sample in the image, which is equal
        to ``sample_index`` here.

        Raises
        ------
        IndexError
            If ``sample_index`` is greater or equal to the number of patches in the image.
        """
        patches_tensor = self._get_patches(data_point.image.tensor)

        if sample_index >= patches_tensor.size(1):
            raise IndexError(
                f"'sample_index' {sample_index} is out of range as there are only "
                f"{patches_tensor.size(1)} patches in the image."
            )

        return sample_index
