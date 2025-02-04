from pathlib import Path

import numpy as np
import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.data.structures import DataPoint
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


def test_extract_sample():
    patch = Patch(patch_size=(2, 3, 2), stride=(1, 2, 1))
    affine = np.diag([3, 2, 1, 1])
    image_tensor = torch.randn(1, 5, 7, 3)
    mask_1 = torch.ones(1, 5, 7, 3)
    label = torch.ones(1, 5, 7, 3)

    data_point = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor, affine=affine),
        label=tio.LabelMap(tensor=label, affine=affine),
        participant="sub-000",
        session="ses-M000",
        mask_1=tio.LabelMap(tensor=mask_1, affine=affine),
    )
    extracted_data_point, description = patch.extract_sample(data_point, sample_index=5)
    assert description == 5
    assert isinstance(extracted_data_point.image, tio.ScalarImage)
    assert (extracted_data_point.image.tensor == image_tensor[:, :2, 4:7, 1:3]).all()
    assert isinstance(extracted_data_point.label, tio.LabelMap)
    assert (extracted_data_point.label.tensor == label[:, :2, 4:7, 1:3]).all()
    assert isinstance(extracted_data_point.mask_1, tio.LabelMap)
    assert (extracted_data_point.mask_1.tensor == mask_1[:, :2, 4:7, 1:3]).all()

    assert np.isclose(extracted_data_point.image.affine, affine).all()
    assert np.isclose(extracted_data_point.label.affine, affine).all()

    data_point = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor),
        label=1,
        participant="sub-000",
        session="ses-000",
    )
    extracted_data_point, _ = patch.extract_sample(data_point, sample_index=5)
    assert extracted_data_point.label == 1

    with pytest.raises(IndexError):
        patch.extract_sample(data_point, sample_index=25)

    patch = Patch(patch_size=2, stride=1)
    extracted_data_point, description = patch.extract_sample(data_point, sample_index=1)
    assert (
        extracted_data_point.image.tensor == image_tensor[:, :2, :2, 1:3]
    ).all()  # .view starts with the last dimension


def test_format_output():
    patch = Patch(patch_size=(3, 4, 3), stride=2)
    affine = np.diag([3, 2, 1, 1])
    image_tensor = torch.randn(1, 3, 4, 5)
    mask_1 = torch.ones(1, 3, 4, 5)
    label = torch.ones(1, 3, 4, 5)

    sample_data = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor, affine=affine),
        label=tio.LabelMap(tensor=label, affine=affine),
        mask_1=tio.LabelMap(tensor=mask_1, affine=affine),
        participant="sub-000",
        session="ses-M000",
    )
    output = patch.format_output(
        sample_data,
        image_path=Path("sub-000_ses-M000_T1w.nii.gz"),
        description=1,
    )
    assert (output.sample == image_tensor).all()
    assert (output.label == label).all()
    assert np.isclose(output.affine, affine).all()
    assert output.session == "ses-M000"
    assert output.participant == "sub-000"
    assert output.extraction == "patch"
    assert output.image_path == "sub-000_ses-M000_T1w.nii.gz"
    assert output.patch_index == 1
    assert output.patch_size == (3, 4, 3)
    assert output.patch_stride == (2, 2, 2)

    sample_data = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor),
        participant="sub-000",
        session="ses-M000",
        label=0.5,
    )
    output = patch.format_output(
        sample_data,
        image_path=Path("sub-000_ses-M000_T1w.nii.gz"),
        description=1,
    )
    assert output.label == 0.5
