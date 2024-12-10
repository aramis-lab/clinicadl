import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from clinicadl.transforms.extraction import Image


def test_extract_method():
    image = Image()
    assert image.extract_method == "image"


def test_num_samples_per_image():
    image = Image()
    assert image.num_samples_per_image(torch.randn(1, 3, 4, 5)) == 1


def test_sample_path():
    image = Image()
    assert image.sample_path(
        Path("sub-001/ses-M000/sub-001_ses-M000_T1w.nii.gz"), 0
    ) == Path("sub-001/ses-M000/sub-001_ses-M000_T1w.pt")
    assert image.sample_path(
        Path("sub-001/ses-M001/sub-001_ses-M001_FLAIR.nii"), 0
    ) == Path("sub-001/ses-M001/sub-001_ses-M001_FLAIR.pt")


def test_extract_sample():
    image = Image()
    image_tensor = torch.randn(1, 3, 4, 5)
    assert (image.extract_sample(image_tensor, sample_index=0) == image_tensor).all()

    with pytest.raises(IndexError):
        image.extract_sample(image_tensor, sample_index=1)


def test_extract():
    tmp_dir = Path(__file__).parents[2] / "ressources" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    image_tensor = torch.randn(1, 3, 4, 5)
    image_nifti = nib.Nifti1Image(image_tensor.squeeze(0).numpy(), np.eye(4))
    nib.save(image_nifti, tmp_dir / "sub-001_ses-M000_T1w.nii.gz")
    nib.save(image_nifti, tmp_dir / "sub-001_ses-M001_FLAIR.nii")

    image = Image()
    output = image.extract(tmp_dir / "sub-001_ses-M000_T1w.nii.gz")
    assert len(output) == 1
    assert output[0][0] == tmp_dir / "sub-001_ses-M000_T1w.pt"
    assert (output[0][1] == image_tensor).all()

    output = image.extract(tmp_dir / "sub-001_ses-M001_FLAIR.nii")
    assert output[0][0] == tmp_dir / "sub-001_ses-M001_FLAIR.pt"
    assert (output[0][1] == image_tensor).all()

    shutil.rmtree(tmp_dir)
