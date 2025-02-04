from pathlib import Path

import numpy as np
import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.data.structures import DataPoint
from clinicadl.transforms.extraction import Slice


def test_args():
    with pytest.raises(ValidationError):
        Slice(slices=[0], slice_direction=3)
    with pytest.raises(ValidationError):
        Slice(slices=[0], discarded_slices=[1])
    with pytest.raises(ValidationError):
        Slice(slices=[0], borders=1)


def test_extract_method():
    slice = Slice(slices=[0, 1, 2])
    assert slice.extract_method == "slice"


def test_num_samples_per_image():
    img = torch.randn(1, 5, 7, 3)

    slice = Slice()
    assert slice.num_samples_per_image(img) == 5

    slice = Slice(slices=[1, 2])
    assert slice.num_samples_per_image(img) == 2

    slice = Slice(borders=2, slice_direction=1)
    assert slice.num_samples_per_image(img) == 3

    slice = Slice(discarded_slices=[1, 2], slice_direction=2)
    assert slice.num_samples_per_image(img) == 1

    slice = Slice(discarded_slices=[1], borders=2, slice_direction=0)
    assert slice.num_samples_per_image(img) == 1

    slice = Slice(discarded_slices=[2], borders=2, slice_direction=0)
    assert slice.num_samples_per_image(img) == 0

    slice = Slice(discarded_slices=[3], slice_direction=2)
    with pytest.raises(IndexError):
        slice.num_samples_per_image(img)

    slice = Slice(slices=[3], slice_direction=2)
    with pytest.raises(IndexError):
        slice.num_samples_per_image(img)


def test_extract_sample():
    slice = Slice(discarded_slices=[4], borders=1, slice_direction=2)
    affine = np.diag([3, 2, 1, 1])
    image_tensor = torch.randn(1, 5, 3, 7)
    mask_1 = torch.ones(1, 5, 3, 7)
    label = torch.ones(1, 5, 3, 7)

    data_point = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor, affine=affine),
        label=tio.LabelMap(tensor=label, affine=affine),
        participant="sub-000",
        session="ses-M000",
        mask_1=tio.LabelMap(tensor=mask_1, affine=affine),
    )
    extracted_data, description = slice.extract_sample(data_point, sample_index=3)
    assert description == 5
    assert isinstance(extracted_data.image, tio.ScalarImage)
    assert (extracted_data.image.tensor == image_tensor[:, :, :, 5:6]).all()
    assert isinstance(extracted_data.label, tio.LabelMap)
    assert (extracted_data.label.tensor == label[:, :, :, 5:6]).all()
    assert isinstance(extracted_data.mask_1, tio.LabelMap)
    assert (extracted_data.mask_1.tensor == mask_1[:, :, :, 5:6]).all()

    assert np.isclose(extracted_data.image.affine, affine).all()
    assert np.isclose(extracted_data.label.affine, affine).all()

    data_point = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor),
        participant="sub-000",
        session="ses-M000",
        label=1,
    )
    extracted_data, _ = slice.extract_sample(data_point, sample_index=1)
    assert extracted_data.label == 1

    with pytest.raises(IndexError):
        slice.extract_sample(data_point, sample_index=4)

    slice = Slice(slices=[2, 3])
    assert (
        slice.extract_sample(data_point, sample_index=1)[0].image.tensor
        == image_tensor[:, 3:4]
    ).all()

    slice = Slice(discarded_slices=[0], slice_direction=1)
    assert (
        slice.extract_sample(data_point, sample_index=0)[0].image.tensor
        == image_tensor[:, :, 1:2]
    ).all()


def test_format_output():
    slice = Slice(slice_direction=2, squeeze=True)
    affine = np.diag([3, 2, 1, 1])
    image_tensor = torch.randn(1, 3, 4, 5)
    mask_1 = torch.ones(1, 3, 4, 5)
    label = torch.ones(1, 3, 4, 5)

    sample_data = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor, affine=affine),
        label=tio.LabelMap(tensor=label, affine=affine),
        participant="sub-000",
        session="ses-M000",
        mask_1=tio.LabelMap(tensor=mask_1, affine=affine),
    )
    output = slice.format_output(
        sample_data,
        image_path=Path("sub-000_ses-M000_T1w.nii.gz"),
        description=1,
    )
    assert (output.sample == image_tensor.squeeze(3)).all()
    assert (output.label == label.squeeze(3)).all()
    assert np.isclose(output.affine, affine).all()
    assert output.session == "ses-M000"
    assert output.participant == "sub-000"
    assert output.extraction == "slice"
    assert output.image_path == "sub-000_ses-M000_T1w.nii.gz"
    assert output.slice_direction == 2
    assert output.slice_position == 1

    slice = Slice(slice_direction=2, squeeze=False)
    sample_data = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor),
        participant="sub-000",
        session="ses-M000",
        label=0.5,
    )
    output = slice.format_output(
        sample_data,
        image_path=Path("sub-000_ses-M000_T1w.nii.gz"),
        description=1,
    )
    assert (output.sample == image_tensor.squeeze(3)).all()
    assert output.label == 0.5
