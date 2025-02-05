import copy
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
import torchio as tio

from clinicadl.dictionary.suffixes import PT
from clinicadl.dictionary.words import AFFINE, MASK
from clinicadl.utils.typing import PathType

LabelType = Optional[Union[int, float, tio.LabelMap]]


class Column(str):
    """
    Dummy class to store label when it represents a column of a dataframe.
    """


class DataPoint(tio.Subject):
    """
    Object that gathers an image, the associated label, and any
    mask associated to the image.

    Parameters
    ----------
    image : Union[tio.ScalarImage, PathType]
        the image, as a TorchIO ScalarImage or a path to a nifti file.
    label : Optional[Union[float, int, tio.LabelMap, PathType]]
        the label associated to the image. Can be a float (e.g. regression),
        an int (e.g. classification), a mask (passed as a TorchIO LabelMap
        or a path to a nifti file; e.g. segmentation) or None if no label (e.g. reconstruction).
    **masks : Union[tio.LabelMap, PathType]
        any mask related to the image and useful to compute transforms.
    """

    image: tio.ScalarImage
    label: LabelType
    participant: str
    session: str

    def __init__(
        self,
        image: Union[tio.ScalarImage, PathType],
        label: Optional[Union[float, int, tio.LabelMap, PathType]],
        participant: str,
        session: str,
        **masks: Union[tio.LabelMap, PathType],
    ) -> None:
        if isinstance(image, PathType):
            image = tio.ScalarImage(path=image)

        if isinstance(label, PathType):
            label = tio.LabelMap(path=label)

        for name, mask in masks.items():
            if isinstance(mask, PathType):
                masks[name] = tio.LabelMap(path=mask)

        super().__init__(
            image=image, label=label, participant=participant, session=session, **masks
        )

    @property
    def affine(self):
        """Return affine matrix of first image in subject.

        Consistency of matrices across images in the subject is checked first.
        """
        self.check_consistent_affine()
        return self.get_first_image().affine

    def add_mask(self, mask: Union[tio.LabelMap, PathType], mask_name: str) -> None:
        """
        To add a mask to the DataPoint.

        Parameters
        ----------
        mask : Union[tio.LabelMap, PathType]
            the mask to add, as a TorchIO ScalarImage or a path to a nifti file.
        mask_name : str
            the name that the mask will take in the DataPoint.
        """
        if isinstance(mask, PathType):
            mask = tio.LabelMap(path=mask)
        self.add_image(mask, mask_name)

    def __copy__(self):
        return _subject_copy_helper(self, type(self))


def _subject_copy_helper(
    old_obj: tio.Subject,
    new_subj_cls: Callable[[Dict[str, Any]], tio.Subject],
):
    """
    Adapted torchio.data.subject._subject_copy_helper to work
    with DataPoint.
    """
    result_dict = {}
    for key, value in old_obj.items():
        if isinstance(value, tio.Image):
            value = copy.copy(value)
        else:
            value = copy.deepcopy(value)
        result_dict[key] = value

    new = new_subj_cls(**result_dict)
    new.applied_transforms = old_obj.applied_transforms[:]
    return new


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
    `clinicadl.TensorConversion.save_mask_as_tensor`.\n
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

    def __init__(self, mask: Union[str, PathType]) -> None:
        if isinstance(mask, Path):
            if not self._check_path(mask):
                raise FileNotFoundError(
                    f"The mask has been passed as a Path object (got {mask}), but no such file exists."
                )
            self.is_common_mask = True
            self.path = Path(mask)
            self.name = self.path.name

        elif isinstance(mask, str):
            if self._check_path(mask):
                self.is_common_mask = True
                self.path = Path(mask)
                self.name = self.path.name
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
        See also: :py:func:`clinicadl.data.tensor_conversion.TensorConversion.save_mask_as_tensor`.
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
        filename : Optional[PathType], (optional, default=None)
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
