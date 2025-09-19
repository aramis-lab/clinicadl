from pathlib import Path

import nibabel as nib
import torch

from .exceptions import ClinicaDLLoadingError


def nifti_to_tensor(path: Path, int_values: bool = False) -> torch.Tensor:
    """
    To load nifti files with nibabel and convert them
    to PyTorch tensors.

    The nifti image is expected to be 3D, and a channel dimension will
    be added so that the output tensor is 4D.

    Parameters
    ----------
    path : Path
        The path to the 3D image.
    int_values : bool, default=False
        Whether to have integer values in the output tensor.

    Returns
    -------
    torch.Tensor
        The image as a tensor, with a channel dimension added. The values
        of the tensor will be `torch.int32` if 'int_values' is True, `torch.float32`
        otherwise.

    Raises
    ------
    FileNotFoundError
        If 'path' in not a file.
    Exception
        Any exception during the loading of the image.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"The path '{path}' does not match any file.")

    try:
        nifti_image = nib.loadsave.load(path)
    except Exception as e:
        raise ClinicaDLLoadingError(
            f"Unable to read the image in {path}. Consider using a nifti file format "
            "('.nii' or '.nii.gz')."
        ) from e

    image_tensor = torch.from_numpy(nifti_image.get_fdata()).unsqueeze(0)  # type: ignore

    if int_values:
        return image_tensor.int()
    return image_tensor.float()


def pt_to_tensor(path: Path, int_values: bool = False) -> torch.Tensor:
    """
    To load PyTorch tensors.

    Parameters
    ----------
    path : Path
        The path to the tensor.
    int_values : bool, default=False
        Whether to have integer values in the output tensor.

    Returns
    -------
    torch.Tensor
        The tensor. The values will be `torch.int32` if 'int_values' is True,
        `torch.float32` otherwise.

    Raises
    ------
    FileNotFoundError
        If 'path' in not a file.
    Exception
        Any exception during the loading of the image.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"The path '{path}' does not match any file.")

    try:
        image_tensor: torch.Tensor = torch.load(path, weights_only=True)
    except Exception as e:
        raise ClinicaDLLoadingError(f"Unable to read the tensor in {path}.") from e

    if int_values:
        return image_tensor.int()
    else:
        return image_tensor.float()
