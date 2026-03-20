from pathlib import Path

import numpy as np
import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.data.structures import DataPoint
from clinicadl.transforms.extraction import Slice
from clinicadl.utils.exceptions import ClinicaDLTSVError

CAPS_DIR = Path(__file__).parents[2] / "resources" / "caps_example"
SLICE_TSV = CAPS_DIR / "tsv" / "extract_slices_test.tsv"
BAD_SLICE_TSV_1 = CAPS_DIR / "tsv" / "extract_slices_test_bad.tsv"


def test_args():
    with pytest.raises(
        ValidationError, match="'slices' and 'tsv_path' can't be passed simultaneously."
    ):
        Slice(slices=[0], tsv_path=SLICE_TSV)
    with pytest.raises(ValidationError):
        Slice(slices=[0], slice_direction=3)
    with pytest.raises(
        ValidationError,
        match="You can't pass 'discarded_slices' if 'slices' or 'tsv_path' was passed.",
    ):
        Slice(slices=[0], discarded_slices=[1])
    with pytest.raises(
        ValidationError,
        match="You can't pass 'borders' if 'slices' or 'tsv_path' was passed.",
    ):
        Slice(tsv_path=SLICE_TSV, borders=1)
    with pytest.raises(
        ClinicaDLTSVError,
        match="TSV must contain columns: 'participant_id', 'session_id', 'slice_idx'",
    ):
        Slice(tsv_path=BAD_SLICE_TSV_1)


def test_num_samples_per_image():
    affine = np.diag([3, 2, 1, 1])
    image_tensor = torch.randn(2, 5, 7, 3)
    label = torch.randint(0, 2, (2, 5, 3, 7))

    data_point = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor, affine=affine),
        label=tio.LabelMap(tensor=label, affine=affine),
        participant="sub-000",
        session="ses-M000",
        image_path="abc.nii.gz",
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

    # test FromTSV
    slice = Slice(tsv_path=SLICE_TSV, slice_direction=1)
    assert slice.num_samples_per_image(data_point) == 2


def test_extract_sample():
    slice = Slice(discarded_slices=[4], borders=1, slice_direction=2)
    affine = np.diag([3, 2, 1, 1])
    image_tensor = torch.randn(1, 5, 3, 7)
    label = torch.randint(0, 2, (2, 5, 3, 7))

    data_point = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor, affine=affine),
        label=tio.LabelMap(tensor=label, affine=affine),
        participant="sub-000",
        session="ses-M000",
        image_path="abc.nii.gz",
    )
    extracted_data = slice(data_point, sample_index=3)
    assert isinstance(extracted_data.image, tio.ScalarImage)
    assert (extracted_data.image.tensor == image_tensor[:, :, :, 5:6]).all()
    assert isinstance(extracted_data.label, tio.LabelMap)
    assert (extracted_data.label.tensor == label[:, :, :, 5:6]).all()

    assert np.isclose(extracted_data.image.affine, affine).all()
    assert np.isclose(extracted_data.label.affine, affine).all()

    assert extracted_data.participant == "sub-000"
    assert extracted_data.session == "ses-M000"
    assert extracted_data["image_path"] == "abc.nii.gz"
    assert extracted_data["slice_direction"] == 2
    assert extracted_data["sample_position"] == 5
    assert extracted_data["sample_type"] == "slice"

    assert data_point.image.tensor.shape == (1, 5, 3, 7)

    # test transforms history
    transform = tio.Clamp(out_min=0, out_max=10)
    sample = slice(transform(data_point), sample_index=0)
    assert len(sample.get_applied_transforms()) == 1
    assert isinstance(sample.get_applied_transforms()[0], tio.Clamp)

    # other tests
    data_point = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor),
        participant="sub-000",
        session="ses-M000",
        label=1,
    )
    extracted_data = slice(data_point, sample_index=1)
    assert extracted_data.label == 1

    with pytest.raises(IndexError):
        slice(data_point, sample_index=4)

    slice = Slice(slices=[2, 3])
    assert (
        slice(data_point, sample_index=1).image.tensor == image_tensor[:, 3:4]
    ).all()

    slice = Slice(discarded_slices=[0], slice_direction=1)
    assert (
        slice(data_point, sample_index=0).image.tensor == image_tensor[:, :, 1:2]
    ).all()

    # generator
    gen = slice(data_point)
    list_sample_indices = [sample["sample_position"] for sample in gen]
    assert list_sample_indices == [1, 2]

    # Test from FromTSV
    extractor = Slice(tsv_path=SLICE_TSV, slice_direction=2)

    data_point = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor, affine=affine),
        label=tio.LabelMap(tensor=label, affine=affine),
        participant="sub-000",
        session="ses-M000",
        image_path="abc.nii.gz",
    )

    extracted_data = extractor(data_point, sample_index=0)

    assert isinstance(extracted_data.image, tio.ScalarImage)
    assert (extracted_data.image.tensor == image_tensor[:, :, :, 1:2]).all()
    assert isinstance(extracted_data.label, tio.LabelMap)
    assert (extracted_data.label.tensor == label[:, :, :, 1:2]).all()

    data_point.session = "ses-M001"
    with pytest.raises(
        ValueError,
        match="No slices found in TSV for participant=sub-000, session=ses-M001.",
    ):
        extractor(data_point, sample_index=0)
