from logging import getLogger
from typing import Tuple, Union

import torch
import torchio as tio
from pydantic import NonNegativeInt, PositiveInt, computed_field

from clinicadl.data.structures import DataPoint
from clinicadl.utils.enum import ExtractionMethod
from clinicadl.utils.typing import PathType

from .base import Extraction, Sample

logger = getLogger("clinicadl.transforms.extraction.patch")


class PatchSample(Sample):
    """
    Output of a CapsDataset when patch extraction is performed.

    Attributes
    ----------
    sample : torch.Tensor
        the patch as a 4D PyTorch tensor (with one channel dimension).
    affine : np.ndarray
        the affine matrix associated to the patch.
    participant : str
        the subject concerned.
    session : str
        the session concerned.
    image_path : str
        the path to the image from which the patch has been extracted.
    label : Optional[Union[float, int, torch.Tensor]]
        the potential label associated to the image.
    patch_index : NonNegativeInt
        the index of the patch among all patches extracted from the image.
    patch_size : Tuple[PositiveInt, PositiveInt, PositiveInt]
        the size of the patch.
    patch_stride : Tuple[PositiveInt, PositiveInt, PositiveInt]
        the stride used for patch extraction.
    """

    patch_index: NonNegativeInt
    patch_size: Tuple[PositiveInt, PositiveInt, PositiveInt]
    patch_stride: Tuple[PositiveInt, PositiveInt, PositiveInt]

    @computed_field
    @property
    def extraction(self) -> str:
        """The extraction method."""
        return ExtractionMethod.PATCH.value


class Patch(Extraction):
    """
    Transform class to extract patches from an image.

    This class enables patches extraction from an image tensor. The image is divided into smaller patches
    using a sliding window approach, where the patch size and the stride are configurable.

    Parameters
    ----------
    patch_size :  Union[PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]] (optional, default=50)
        The size of each patch. If a single value is passed, the same patch size will be used for the three
        spatial dimensions.
    stride : Union[PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]] (optional, default=50)
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

    def extract_sample(
        self, data_point: DataPoint, sample_index: int
    ) -> Tuple[DataPoint, int]:
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
            present in the original `data_point`. The patch extracted from an
            image is accessible via the same name as was the image in the original
            `data_point`.
        int
            The patch index. For compatibility, as the patch index is always equal
            to `sample_index` here.

        Raises
        ------
        IndexError
            If 'sample_index' is greater or equal to the number of patches in the images.
        """
        return super().extract_sample(data_point, sample_index)

    def num_samples_per_image(self, image: torch.Tensor) -> int:
        """
        Returns the total number of patches extracted from an image.

        Parameters
        ----------
        image : torch.Tensor
            The input image tensor (4D), where the first dimension represents the channel dimension.

        Returns
        -------
        int
            The total number of patches that can be extracted from the image.

        Notes
        -----
        The number of patches is determined by the image size, the patch size, and the stride.
        """
        return self._get_patches(image).shape[0]

    def format_output(
        self,
        data_point: DataPoint,
        image_path: PathType,
        description: int,
    ) -> PatchSample:
        """
        Puts all the output information in an PatchSample object.

        Parameters
        ----------
        data_point : DataPoint
            the `DataPoint` object associated to the patch.
        image_path : PathType
            the path of the image from which the patch is extracted.
        description : int
            the patch index.

        Returns
        -------
        PatchSample
            a PatchSample object with the patch and all the relevant information on the patch.
        """
        return PatchSample(
            sample=data_point.image.tensor,
            affine=data_point.image.affine,
            participant=data_point.participant,
            session=data_point.session,
            image_path=str(image_path),
            label=data_point.label.tensor
            if isinstance(data_point.label, tio.LabelMap)
            else data_point.label,
            patch_index=description,
            patch_size=self.patch_size,
            patch_stride=self.stride,
        )

    def _extract_tensor_sample(
        self, image_tensor: torch.Tensor, sample_index: int
    ) -> torch.Tensor:
        """
        Extracts a single patch from an image.

        Raises
        ------
        IndexError
            If 'sample_index' is greater or equal to the number of patches in the image.
        """
        patches_tensor = self._get_patches(image_tensor)
        try:
            return patches_tensor[sample_index].unsqueeze(0).clone()
        except IndexError as exc:
            raise IndexError(
                f"'sample_index' {sample_index} is out of range as there are only "
                f"{len(patches_tensor)} patches in the image."
            ) from exc

    def _get_sample_description(
        self, image_tensor: torch.Tensor, sample_index: int
    ) -> int:
        """The sample description for patch extraction is the index of the patch."""
        return sample_index

    def _get_patches(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Creates a tensor of patches from the image using the PyTorch method `unfold`.

        Returns
        -------
        torch.Tensor
            A tensor containing all the patches extracted from the image. The tensor shape
            will be `(num_patches, patch_size[0], patch_size[1], patch_size[2])`, where `num_patches` is
            determined by the image size, the patch size, and the stride.
        """
        patches_tensor = (
            image_tensor.unfold(1, self.patch_size[0], self.stride[0])
            .unfold(2, self.patch_size[1], self.stride[1])
            .unfold(3, self.patch_size[2], self.stride[2])
            .contiguous()
        )

        return patches_tensor.view(
            -1, self.patch_size[0], self.patch_size[1], self.patch_size[2]
        )
