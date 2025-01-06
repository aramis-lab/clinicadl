import copy
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
import torchio as tio

from clinicadl.dictionary.words import LABEL


class DataPoint(tio.Subject):
    """
    Object that gathers an image, the associated label, and any
    mask associated to the image.

    Parameters
    ----------
    image : Union[torch.Tensor, tio.ScalarImage]
        the image, as a Pytorch tensor or a TorchIO ScalarImage.
    label : Optional[Union[float, int, torch.Tensor, tio.LabelMap]]
        the label associated to the image. Can be a float (e.g. regression),
        an int (e.g. classification), a mask (passed as torch Tensor or a
        TorchIO LabelMap; e.g. segmentation) or None if no label (e.g. reconstruction).
    **masks : Union[torch.Tensor, tio.LabelMap]
        any mask related to the image and useful to compute transforms.

    Raises
    ------
    AssertionError
        If all the images/masks passed don't have the same shape.
    """

    image: tio.ScalarImage
    label: Optional[Union[float, int, tio.LabelMap]]

    def __init__(
        self,
        image: Union[torch.Tensor, tio.ScalarImage],
        label: Optional[Union[float, int, torch.Tensor, tio.LabelMap]],
        **masks: Union[torch.Tensor, tio.LabelMap],
    ) -> None:
        if not isinstance(image, tio.ScalarImage):
            image = tio.ScalarImage(tensor=image)
        image_shape = image.tensor.shape

        if isinstance(label, torch.Tensor):
            label = tio.LabelMap(tensor=label)

        for name, mask in masks.items():
            if not isinstance(mask, tio.LabelMap):
                masks[name] = tio.LabelMap(tensor=mask)
        masks[LABEL] = label

        for name, mask in masks.items():
            if isinstance(mask, tio.LabelMap):
                assert mask.tensor.shape == image_shape, (
                    f"Masks must be the same shape as the image, but got "
                    f"{image_shape} for the image and {mask.tensor.shape} "
                    f"for '{name}')"
                )

        super().__init__(image=image, **masks)

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
    between a mask passed as a file name, that corresponds to a common mask,
    and a mask passed as a suffix (a simple string), that corresponds to a mask
    specific to each subject.

    For example, `Mask("masks/mask.nii.gz")` will be understood has a common
    mask, where as `Mask("mask")` will be understood has a specific mask.

    In the latter case, it is expected that all the (subject, session) studied
    have the associated mask in their CAPS folders. It will look for files with
    the suffix `mask` in these folders.

    Parameters
    ----------
    filename : Union[str, Path]
        the mask, passed as a path or a suffix.
    """

    def __init__(self, mask: Union[str, Path]) -> None:
        if isinstance(mask, Path):
            if not self._check_path(mask):
                raise ValueError(
                    f"The mask has been passed as a Path object (got {mask}), but no such file exists."
                )
            self.common_mask = True
            self.mask = Path(mask)

        elif isinstance(mask, str):
            if self._check_path(mask):
                self.common_mask = True
                self.mask = Path(mask)
            else:
                self.common_mask = False
                self.mask = mask

    @staticmethod
    def _check_path(mask_path: Union[str, Path]) -> bool:
        """Checks if the mask file exists."""
        mask_path = Path(mask_path)
        return mask_path.is_file()

    def get_associated_mask(self, filename: Union[str, Path]) -> Path:
        """
        Returns the mask associated to an image.

        If the mask is common to all subjects and sessions, the method will
        simply return it. On the other hand, if the mask is specific to each
        (subject, session), the method will use the input `filename` to get
        the associated mask.

        Parameters
        ----------
        filename : Union[str, Path]
            the image whose associated mask is to be found.

        Returns
        -------
        Path :
            the path to the mask associated to the image.

        Raises
        ------
        ValueError
            if the associated mask doesn't exist.

        Examples
        --------
        >>> mask=Mask("seg")
        >>> mask.get_associated_mask("sub-001_ses-M000_T1w.nii.gz")
        PosixPath('sub-001_ses-M000_seg.nii.gz')

        >>> mask=Mask("masks/leftHippocampus.nii.gz")
        >>> mask.get_associated_mask("sub-001_ses-M000_T1w.nii.gz")
        PosixPath('masks/leftHippocampus.nii.gz')
        """

        if self.common_mask:
            return self.mask
        else:
            filename = Path(filename)
            without_extension = str(filename).rstrip("".join(filename.suffixes))
            suffix = without_extension.rsplit("_", maxsplit=1)[-1]
            mask_file = str(filename).replace(f"_{suffix}.", f"_{self.mask}.")
            if not self._check_path(mask_file):
                raise ValueError(
                    f"A mask associated to {str(filename)} was expected "
                    f"to be found in {mask_file}, but there is no such file."
                )

            return Path(mask_file)
