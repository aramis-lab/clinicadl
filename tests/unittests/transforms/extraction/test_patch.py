import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from pydantic import ValidationError

from clinicadl.transforms.extraction import Patch


def test_args():
    with pytest.raises(ValidationError):
        Patch(patch_size=0)
    with pytest.raises(ValidationError):
        Patch(stride=0)


def test_extract_method():
    patch = Patch()
    assert patch.extract_method == "patch"


def test_num_samples_per_image():
    img = torch.randn(1, 5, 7, 3)

    patch = Patch(patch_size=3, stride=1)
    assert patch.num_samples_per_image(img) == 3 * 5 * 1

    patch = Patch(patch_size=(2, 3, 2), stride=(1, 2, 1))
    assert patch.num_samples_per_image(img) == 4 * 3 * 2

    patch = Patch(patch_size=(2, 3, 2), stride=3)
    assert patch.num_samples_per_image(img) == 2 * 2 * 1


def test_sample_path():
    patch = Patch(patch_size=(2, 3, 2), stride=3)
    assert patch.sample_path(
        Path("sub-001/ses-M000/sub-001_ses-M000_T1w.nii.gz"), 3
    ) == Path(
        "sub-001/ses-M000/sub-001_ses-M000_patchsize-2x3x2_stride-3x3x3_patch-3_T1w.pt"
    )

    patch = Patch(patch_size=(2, 3, 2), stride=(1, 2, 1))
    assert patch.sample_path(
        Path("sub-001/ses-M001/sub-001_ses-M001_FLAIR.nii"), 7
    ) == Path(
        "sub-001/ses-M001/sub-001_ses-M001_patchsize-2x3x2_stride-1x2x1_patch-7_FLAIR.pt"
    )


def test_extract_sample():
    image_tensor = torch.randn(1, 5, 7, 3)

    patch = Patch(patch_size=2, stride=1)
    assert (
        patch.extract_sample(image_tensor, sample_index=1)
        == image_tensor[:, :2, :2, 1:3]  # .view starts with the last dimension
    ).all()

    patch = Patch(patch_size=(2, 3, 2), stride=(1, 2, 1))
    assert (
        patch.extract_sample(image_tensor, sample_index=5)
        == image_tensor[:, :2, 4:7, 1:3]
    ).all()

    patch = Patch(patch_size=(2, 3, 2), stride=(1, 2, 1))
    assert (
        patch.extract_sample(image_tensor, sample_index=7)
        == image_tensor[:, 1:3, :3, 1:3]
    ).all()

    patch = Patch(patch_size=(2, 3, 2), stride=(1, 2, 1))
    with pytest.raises(IndexError):
        patch.extract_sample(image_tensor, sample_index=24)


def test_extract():
    tmp_dir = Path(__file__).parents[2] / "ressources" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    image_tensor = torch.randn(1, 5, 7, 4)
    image_nifti = nib.Nifti1Image(image_tensor.squeeze(0).numpy(), np.eye(4))
    nib.save(image_nifti, tmp_dir / "sub-001_ses-M000_T1w.nii.gz")

    image = Patch(patch_size=2, stride=2)
    output = image.extract(tmp_dir / "sub-001_ses-M000_T1w.nii.gz")
    assert len(output) == 12
    assert (
        output[0][0]
        == tmp_dir / "sub-001_ses-M000_patchsize-2x2x2_stride-2x2x2_patch-0_T1w.pt"
    )
    assert (output[0][1] == image_tensor[:, :2, :2, :2]).all()
    assert (
        output[3][0]
        == tmp_dir / "sub-001_ses-M000_patchsize-2x2x2_stride-2x2x2_patch-3_T1w.pt"
    )
    assert (output[3][1] == image_tensor[:, :2, 2:4, 2:4]).all()

    shutil.rmtree(tmp_dir)
