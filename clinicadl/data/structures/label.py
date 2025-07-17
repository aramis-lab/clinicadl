from collections import UserString
from pathlib import Path
from typing import Optional, Union

import torch
import torchio as tio

from clinicadl.dictionary.suffixes import PT
from clinicadl.dictionary.words import AFFINE, MASK
from clinicadl.utils.typing import PathType

LabelType = Optional[Union[int, float, dict[str, float], tio.LabelMap]]


class Column(UserString):
    """
    Dummy class to store label when it represents a column of a dataframe.
    """

    def __init__(self, name: str):
        self._name = name
        super().__init__(name)

    def __str__(self):
        return f"Column('{self._name}')"


class Mask:
    """To handle masks in ClinicaDL. More precisely, it makes the difference
    between a mask passed as an image path, that corresponds to a common mask,
    and a mask passed as a suffix (a simple string), that corresponds to a mask
    specific to each image.

    For example, `Mask("masks/mask.nii.gz")` will be understood has a common
    mask, whereas `Mask("mask")` will be understood has an image-specific mask.

    In the latter case, it is expected that all the images studied
    have the associated mask in the CAPS directory.

    If the mask is in a `.pt` file (e.g. `Mask("masks/mask.pt")`), it is expected
    to be a 4D tensor with the associated affine matrix, as saved by
    `clinicadl.TensorConversion._save_mask_as_tensor`.\n
    If the mask is in a NIfTI file (e.g. `Mask("masks/mask.nii.gz")`), it is expected
    to be a 3D image.

    Parameters
    ----------
    mask : mask
        the mask, passed as a path or a suffix.

    Raises
    ------
    FileNotFoundError
        if `mask` is passed as a path that does not match any file.
    """

    def __init__(self, mask: PathType) -> None:
        if isinstance(mask, Path):
            if not self._check_path(mask):
                raise FileNotFoundError(
                    f"The mask has been passed as a Path object (got {mask}), but no such file exists."
                )
            self.is_common_mask = True
            self.path = Path(mask)
            self.name = self.path.with_suffix(
                ""
            ).stem  # with_suffix to handle double extensions

        elif isinstance(mask, str):
            if self._check_path(mask):
                self.is_common_mask = True
                self.path = Path(mask)
                self.name = self.path.with_suffix("").stem
            else:
                self.is_common_mask = False
                self.path = None
                self.name = mask

        self._mask_img: Union[tio.LabelMap, None] = None  # lazy loading

    @staticmethod
    def _check_path(mask_path: PathType) -> bool:
        """Checks if the mask file exists."""
        mask_path = Path(mask_path)
        return mask_path.is_file()

    def __str__(self):
        if self.is_common_mask:
            return f"Mask('{self.path}')"
        else:
            return f"Mask('{self.name}')"

    @classmethod
    def _load_mask(cls, path: Path) -> tio.LabelMap:
        """
        Loads a mask (in nifti or .pt file) and return a TorchIO LabelMap.
        """
        if path.suffix == PT:
            mask_tensor, affine = cls._load_pt_mask(path)
            return tio.LabelMap(tensor=mask_tensor, affine=affine)
        else:
            return tio.LabelMap(path=path)

    @staticmethod
    def _load_pt_mask(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Loads a mask and its affine matrix from a .pt file.
        See also: :py:func:`clinicadl.data.tensor_conversion.TensorConversion._save_mask_as_tensor`.
        """
        pt_mask = torch.load(path, weights_only=True)
        return pt_mask[MASK], pt_mask[AFFINE]

    def _lazy_load_common_mask(self) -> tio.LabelMap:
        """
        Gets or loads a common mask (in nifti or .pt file).
        """
        if self._mask_img is None:
            self._mask_img = self._load_mask(self.path)
        return self._mask_img

    def _get_associated_mask_path(self, filename: Path) -> Path:
        """
        Returns the path of the mask associated to an image, when the
        mask is not a common mask.

        Examples
        --------
        >>> mask=Mask("brain")
        >>> mask._get_associated_mask_path("sub-000_ses-M000_pet.nii.gz")
        sub-000_ses-M000_brain.nii.gz
        """
        suffix = str(filename.with_suffix("").stem).rsplit("_", maxsplit=1)[
            -1
        ]  # with_suffix to handle double extensions
        mask_file = str(filename).replace(f"_{suffix}.", f"_{self.name}.")

        return Path(mask_file)

    def get_associated_mask(self, filename: Optional[PathType] = None) -> tio.LabelMap:
        """
        Returns the mask associated to an image, in a TorchIO LabelMap.

        If the mask is common to all subjects and sessions, the method will
        simply return it. On the other hand, if the mask is specific to each
        image, the method will use the input `filename` to get
        the associated mask.

        Parameters
        ----------
        filename : Optional[PathType], default=None
            the image whose associated mask is to be found.
            Can be None if the mask is a common mask (thus it does not depend
            on 'filename').

        Returns
        -------
        tio.LabelMap :
            the mask, in a TorchIO LabelMap.

        Raises
        ------
        FileNotFoundError
            if the associated mask doesn't exist.

        Examples
        --------
        >>> mask=Mask("seg")
        >>> mask.get_associated_mask_path("sub-001_ses-M000_T1w.nii.gz")
        # will get the image in 'sub-001_ses-M000_seg.nii.gz'

        >>> mask=Mask("masks/leftHippocampus.nii.gz")
        >>> mask.get_associated_mask_path("sub-001_ses-M000_T1w.nii.gz")
        # will get the image in 'masks/leftHippocampus.nii.gz'
        >>> mask.get_associated_mask_path()
        # will get the image in 'masks/leftHippocampus.nii.gz'
        """
        if self.is_common_mask:
            return self._lazy_load_common_mask()
        else:
            if filename is None:
                raise ValueError(
                    f"The mask {self.name} is an image-specific mask, "
                    "you must therefore give a 'filename' to get the associated "
                    "mask."
                )

            filename = Path(filename)
            mask_file = self._get_associated_mask_path(filename)
            try:
                return self._load_mask(mask_file)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"No file matches {self.name}, so it is understood as a suffix. "
                    f"Therefore, the mask associated to {str(filename)} was expected "
                    f"to be found in {str(mask_file)}, but there is no such file."
                ) from exc
