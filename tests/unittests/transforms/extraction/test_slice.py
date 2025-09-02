from pathlib import Path

import numpy as np
import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.data.structures import DataPoint
from clinicadl.transforms.extraction import Slice, SliceFromTSV


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
    affine = np.diag([3, 2, 1, 1])
    image_tensor = torch.randn(2, 5, 7, 3)
    mask_1 = torch.randint(0, 2, (1, 5, 3, 7))
    label = torch.randint(0, 2, (2, 5, 3, 7))

    data_point = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor, affine=affine),
        label=tio.LabelMap(tensor=label, affine=affine),
        participant="sub-000",
        session="ses-M000",
        image_path="abc.nii.gz",
        mask_1=tio.LabelMap(tensor=mask_1, affine=affine),
    )

    slice = Slice()
    assert slice.num_samples_per_image(data_point) == 5

    slice = Slice(slices=[1, 2])
    assert slice.num_samples_per_image(data_point) == 2

    slice = Slice(borders=2, slice_direction=1)
    assert slice.num_samples_per_image(data_point) == 3

    slice = Slice(discarded_slices=[1, 2], slice_direction=2)
    assert slice.num_samples_per_image(data_point) == 1

    slice = Slice(discarded_slices=[1], borders=2, slice_direction=0)
    assert slice.num_samples_per_image(data_point) == 1

    slice = Slice(discarded_slices=[2], borders=2, slice_direction=0)
    assert slice.num_samples_per_image(data_point) == 0

    slice = Slice(discarded_slices=[3], slice_direction=2)
    with pytest.raises(IndexError):
        slice.num_samples_per_image(data_point)

    slice = Slice(slices=[3], slice_direction=2)
    with pytest.raises(IndexError):
        slice.num_samples_per_image(data_point)

    # test SliceFromTSV
    caps_dir = Path(__file__).parents[2] / "resources" / "caps_example"
    test_slice_tsv = caps_dir / "tsv" / "extract_slice_test.tsv"

    slice = SliceFromTSV(
        tsv_path=test_slice_tsv, slice_direction=1, one_row_per_slice_mode=True
    )
    assert slice.num_samples_per_image(data_point) == 1

    test_slice_tsv = caps_dir / "tsv" / "extract_slices_not_uniform_test.tsv"

    slice = SliceFromTSV(
        tsv_path=test_slice_tsv, slice_direction=1, one_row_per_slice_mode=False
    )
    assert slice.num_samples_per_image(data_point) == 2

    slice = SliceFromTSV(
        tsv_path=test_slice_tsv, slice_direction=1, one_row_per_slice_mode=True
    )
    assert slice.num_samples_per_image(data_point) == 1


def test_extract_sample():
    slice = Slice(discarded_slices=[4], borders=1, slice_direction=2)
    affine = np.diag([3, 2, 1, 1])
    image_tensor = torch.randn(1, 5, 3, 7)
    mask_1 = torch.randint(0, 2, (1, 5, 3, 7))
    label = torch.randint(0, 2, (2, 5, 3, 7))

    data_point = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor, affine=affine),
        label=tio.LabelMap(tensor=label, affine=affine),
        participant="sub-000",
        session="ses-M000",
        image_path="abc.nii.gz",
        mask_1=tio.LabelMap(tensor=mask_1, affine=affine),
    )
    extracted_data = slice.extract_sample(data_point, sample_index=3)
    assert isinstance(extracted_data.image, tio.ScalarImage)
    assert (extracted_data.image.tensor == image_tensor[:, :, :, 5:6]).all()
    assert isinstance(extracted_data.label, tio.LabelMap)
    assert (extracted_data.label.tensor == label[:, :, :, 5:6]).all()
    assert isinstance(extracted_data["mask_1"], tio.LabelMap)
    assert (extracted_data["mask_1"].tensor == mask_1[:, :, :, 5:6]).all()

    assert np.isclose(extracted_data.image.affine, affine).all()
    assert np.isclose(extracted_data.label.affine, affine).all()

    assert extracted_data.participant == "sub-000"
    assert extracted_data.session == "ses-M000"
    assert extracted_data.image_path == "abc.nii.gz"
    assert extracted_data.slice_position == 5
    assert extracted_data.slice_direction == 2
    assert extracted_data._sample_index == 5

    assert data_point.image.tensor.shape == (1, 5, 3, 7)

    # test transforms history
    transform = tio.Clamp(out_min=0, out_max=10)
    sample = slice.extract_sample(transform(data_point), sample_index=0)
    assert len(sample.get_applied_transforms()) == 1
    assert isinstance(sample.get_applied_transforms()[0], tio.Clamp)

    # other tests
    data_point = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor),
        participant="sub-000",
        session="ses-M000",
        label=1,
    )
    extracted_data = slice.extract_sample(data_point, sample_index=1)
    assert extracted_data.label == 1

    with pytest.raises(IndexError):
        slice.extract_sample(data_point, sample_index=4)

    slice = Slice(slices=[2, 3])
    assert (
        slice.extract_sample(data_point, sample_index=1).image.tensor
        == image_tensor[:, 3:4]
    ).all()

    slice = Slice(discarded_slices=[0], slice_direction=1)
    assert (
        slice.extract_sample(data_point, sample_index=0).image.tensor
        == image_tensor[:, :, 1:2]
    ).all()

    # Test from SliceFromTSV

    # one_row_per_slice_mode=True -> exactly one slice per subject

    caps_dir = Path(__file__).parents[2] / "resources" / "caps_example"
    test_slice_tsv = caps_dir / "tsv" / "extract_slice_test.tsv"

    extractor = SliceFromTSV(
        tsv_path=test_slice_tsv, slice_direction=2, one_row_per_slice_mode=True
    )

    affine = np.diag([3, 2, 1, 1])
    image_tensor = torch.randn(1, 5, 3, 7)
    mask_1 = torch.randint(0, 2, (1, 5, 3, 7))
    label = torch.randint(0, 2, (2, 5, 3, 7))

    data_point = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor, affine=affine),
        label=tio.LabelMap(tensor=label, affine=affine),
        participant="sub-000",
        session="ses-M000",
        image_path="abc.nii.gz",
        mask_1=tio.LabelMap(tensor=mask_1, affine=affine),
    )

    extracted_data = extractor.extract_sample(data_point, sample_index=0)

    assert isinstance(extracted_data.image, tio.ScalarImage)
    assert (extracted_data.image.tensor == image_tensor[:, :, :, 1:2]).all()
    assert isinstance(extracted_data.label, tio.LabelMap)
    assert (extracted_data.label.tensor == label[:, :, :, 1:2]).all()
    assert isinstance(extracted_data["mask_1"], tio.LabelMap)
    assert (extracted_data["mask_1"].tensor == mask_1[:, :, :, 1:2]).all()

    assert sample.participant == "sub-000"
    assert sample.session == "ses-M000"

    assert np.isclose(extracted_data.image.affine, affine).all()
    assert np.isclose(extracted_data.label.affine, affine).all()

    assert extracted_data.participant == "sub-000"
    assert extracted_data.session == "ses-M000"
    assert extracted_data.image_path == "abc.nii.gz"
    assert extracted_data.slice_position == 1
    assert extracted_data.slice_direction == 2
    assert extracted_data._sample_index == 1

    assert data_point.image.tensor.shape == (1, 5, 3, 7)