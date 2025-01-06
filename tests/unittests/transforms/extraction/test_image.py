import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import torchio as tio

from clinicadl.data.structures import DataPoint
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


def test_extract_tensor_sample():
    image = Image()
    image_tensor = torch.randn(1, 3, 4, 5)
    assert (
        image.extract_tensor_sample(image_tensor, sample_index=0) == image_tensor
    ).all()

    with pytest.raises(IndexError):
        image.extract_tensor_sample(image_tensor, sample_index=1)


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


def test_extract_sample():
    image = Image()
    image_tensor = torch.randn(1, 3, 4, 5)
    mask_1 = torch.ones(1, 3, 4, 5)
    mask_2 = torch.zeros(1, 3, 4, 5)
    label = torch.ones(1, 3, 4, 5)

    data_point = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor),
        label=tio.LabelMap(tensor=label),
        mask_1=tio.LabelMap(tensor=mask_1),
        mask_2=tio.LabelMap(tensor=mask_2),
    )
    extracted_data_point, description = image.extract_sample(data_point)
    assert description is None
    assert isinstance(extracted_data_point.image, tio.ScalarImage)
    assert (extracted_data_point.image.tensor == image_tensor).all()
    assert isinstance(extracted_data_point.label, tio.LabelMap)
    assert (extracted_data_point.label.tensor == label).all()
    assert isinstance(extracted_data_point.mask_1, tio.LabelMap)
    assert (extracted_data_point.mask_1.tensor == mask_1).all()
    assert isinstance(extracted_data_point.mask_2, tio.LabelMap)
    assert (extracted_data_point.mask_2.tensor == mask_2).all()

    data_point = DataPoint(image=tio.ScalarImage(tensor=image_tensor), label=1)
    extracted_data_point, _ = image.extract_sample(data_point)
    assert extracted_data_point.label == 1

    with pytest.raises(IndexError):
        image.extract_sample(data_point, sample_index=1)


def test_format_output():
    image = Image()
    image_tensor = torch.randn(1, 3, 4, 5)
    mask_1 = torch.ones(1, 3, 4, 5)
    label = torch.ones(1, 3, 4, 5)

    sample_data = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor),
        label=tio.LabelMap(tensor=label),
        mask_1=tio.LabelMap(tensor=mask_1),
    )
    output = image.format_output(
        sample_data,
        participant_id="sub-001",
        session_id="ses-M001",
        image_path=Path("sub-001_ses-M001_T1w.nii.gz"),
    )
    assert (output.sample == image_tensor).all()
    assert (output.label == label).all()
    assert output.session_id == "ses-M001"
    assert output.participant_id == "sub-001"
    assert output.extraction == "image"
    assert output.image_path == "sub-001_ses-M001_T1w.nii.gz"

    sample_data = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor),
        label=0.5,
    )
    output = image.format_output(
        sample_data,
        participant_id="sub-001",
        session_id="ses-M001",
        image_path=Path("sub-001_ses-M001_T1w.nii.gz"),
    )
    assert output.label == 0.5
