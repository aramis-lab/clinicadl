from logging import getLogger
from pathlib import Path
from typing import List, Tuple, Union

import torch
import torchio as tio
from pydantic import NonNegativeInt, PositiveInt, computed_field, field_validator

from clinicadl.dictionary.suffixes import PT
from clinicadl.utils.enum import ExtractionMethod
from clinicadl.utils.loading import nifti_to_tensor
from clinicadl.utils.typing import PathType

from .base import Extraction, Sample

logger = getLogger("clinicadl.extraction.patch")


class PatchSample(Sample):
    """
    Output of a CapsDataset when patch extraction is performed.

    Attributes
    ----------
    sample : torch.Tensor
        the patch as 4D PyTorch tensor (with one channel dimension).
    participant_id : str
        the subject concerned.
    session_id : str
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
    def extraction(self) -> ExtractionMethod:
        """The extraction method."""
        return ExtractionMethod.PATCH


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

    patch_size: Union[PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]] = 50
    stride: Union[PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]] = 50

    @computed_field
    @property
    def extract_method(self) -> ExtractionMethod:
        """The method to be used for the extraction process (Image, Patch, Slice)."""
        return ExtractionMethod.PATCH

    @field_validator("patch_size", "stride", mode="after")
    @classmethod
    def ensure_tuples(
        cls, v: Union[PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]]
    ) -> Tuple[PositiveInt, PositiveInt, PositiveInt]:
        """
        Ensures that 'patch_size' and 'stride' is always a tuple.
        """
        if isinstance(v, int):
            return (v, v, v)
        else:
            return v

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
        return self.get_patches(image).shape[0]

    def extract(self, nii_path: Path) -> List[Tuple[Path, torch.Tensor]]:
        """
        Extracts all the patches from a NIfTI image tensor.

        Parameters
        ----------
        nii_path : Path
            The path to the NIfTI image file from which patches will be extracted.

        Returns
        -------
        List[Tuple[Path, torch.Tensor]]
            A list of tuples, where each tuple contains an extracted patch,
            and the path where to store it.

        Notes
        -----
        This method extracts patches from the full image and saves them as separate tensors.
        Each patch tensor is returned along with its associated file path.
        """

        image_tensor = nifti_to_tensor(nii_path)
        patches_tensor = self.get_patches(image_tensor)
        patch_list = [
            (self.sample_path(nii_path, idx), patches_tensor[idx])
            for idx in range(patches_tensor.size(0))
        ]
        return patch_list

    def extract_sample(
        self, image_tensor: torch.Tensor, sample_index: int
    ) -> torch.Tensor:
        """
        Extracts a single patch from an image.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The input image tensor from which a patch will be extracted. Must be a 4D tensor
            with a channel dimension and 3 spatial dimensions.
        patch_index : int
            The index of the patch to extract from the image tensor.

        Returns
        -------
        torch.Tensor
            The extracted patch as a 4D tensor (with a channel dimension).

        Raises
        ------
        IndexError
            If 'sample_index' is greater or equal to the number of patches in the image.

        Notes
        -----
        This method allows for the extraction of individual patches based on the provided index.
        """
        patches_tensor = self.get_patches(image_tensor)
        try:
            return patches_tensor[sample_index].unsqueeze(0).clone()
        except IndexError as exc:
            raise IndexError(
                f"'sample_index' {sample_index} is out of range as there are only "
                f"{len(patches_tensor)} patches in the image."
            ) from exc

    def sample_path(self, image_path: Path, sample_index: int) -> Path:
        """
        Constructs the path to save a given patch.

        Parameters
        ----------
        image_path : Path
            The original image path, used to derive the path for saving the patch.
        sample_index : int
            The index of the patch being saved.

        Returns
        -------
        Path
            The path where the patch will be saved, including the patch size and stride
            information in the filename.

        Notes
        -----
        The filename is generated using the original image name, appending patch size, stride,
        and the patch index to ensure each patch is saved with a unique name.
        """
        parent = image_path.parent
        prefix_suffix = image_path.name.rsplit("_", 1)
        patch_size_str = "x".join([str(s) for s in self.patch_size])
        stride_str = "x".join([str(s) for s in self.stride])
        return (
            (
                parent
                / f"{prefix_suffix[0]}_patchsize-{patch_size_str}_stride-{stride_str}_patch-{sample_index}_{prefix_suffix[1]}"
            )
            .with_suffix("")
            .with_suffix(PT)
        )

    def get_patches(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Creates a tensor of patches from the image using the PyTorch method `unfold`.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The input image tensor (4D), where the first dimension represents the channel dimension.

        Returns
        -------
        torch.Tensor
            A tensor containing all the patches extracted from the image. The tensor shape
            will be `(num_patches, patch_size[0], patch_size[1], patch_size[2])`, where `num_patches` is
            determined by the image size, the patch size, and the stride.

        Notes
        -----
        The `unfold` method is used to extract patches by sliding a window over the image.
        The patches are then reshaped into a 4D tensor where each patch is a separate element.
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

    def _get_sample_description(
        self, image_tensor: torch.Tensor, sample_index: int
    ) -> int:
        """The sample description for patch extraction is the index of the patch."""
        return sample_index

    def format_output(
        self,
        tio_sample: tio.Subject,
        participant_id: str,
        session_id: str,
        image_path: PathType,
        description: int,
    ) -> PatchSample:
        """
        Puts all the output information in an PatchSample object.

        Parameters
        ----------
        tio_sample : tio.Subject
            a TorchIO Subject corresponding to the patch, with at least a ScalarImage named 'image'
            and an attribute named 'label'.
        participant_id : str
            the subject concerned.
        session_id : str
            the session concerned.
        image_path : PathType
            the path of the image from which the patch is extracted.
        description : int
            the patch index.

        Returns
        -------
        PatchSample
            a PatchSample object with the patch and all the relevant information on the patch.

        Raises
        ------
        AttributeError
            if `tio_sample` doesn't contain a TorchIO ScalarImage named 'image' and an attribute
            'label'.
        """
        self._check_tio_subject(tio_sample)

        patch = tio_sample.image.tensor
        if isinstance(tio_sample.label, tio.Image):
            label = tio_sample.label.tensor
        else:
            label = tio_sample.label

        return PatchSample(
            sample=patch,
            participant_id=participant_id,
            session_id=session_id,
            image_path=str(image_path),
            label=label,
            patch_index=description,
            patch_size=self.patch_size,
            patch_stride=self.stride,
        )

    def extract_tio_sample(
        self, tio_image: tio.Subject, sample_index: int
    ) -> Tuple[tio.Subject, int]:
        """
        Extracts a patch from a TorchIO Subject.

        Parameters
        ----------
        tio_image : tio.Subject
            The TorchIO Subject to perform extraction on.
        sample_index : int
            Index indicating the patch to extract.

        Returns
        -------
        tio.Subject
            A new TorchIO Subject with the extracted patches for each image
            present in the original `tio_image`. The patch extracted from an
            image is accessible via the same name as was the image in the original
            `tio_image`.
        int
            The patch index. For compatibility, as the patch index is always equal
            to `sample_index` here.

        Raises
        ------
        ValueError
            If all the images in `tio_image` don't have the same shape.
        IndexError
            If 'sample_index' is greater or equal to the number of patches in the images.
        """
        return super().extract_tio_sample(tio_image, sample_index)
