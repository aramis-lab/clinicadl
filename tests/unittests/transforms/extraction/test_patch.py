import numpy as np
import pytest
import torch
import torchio as tio
from monai.inferers import SlidingWindowSplitter
from pydantic import ValidationError

from clinicadl.data.structures import DataPoint
from clinicadl.transforms.extraction import Patch

BAD_INPUTS = [
    {"patch_size": 0},
    {"patch_size": 1, "overlap": 1},
    {"patch_size": 1, "pad_mode": "abc"},
]
GOOD_INPUTS = [
    {"patch_size": 1, "overlap": 0.5, "pad_mode": "reflect"},
    {"patch_size": 1, "pad_mode": "constant"},
    {"patch_size": 1, "pad_mode": "replicate"},
    {"patch_size": 1, "pad_mode": "circular"},
    {"patch_size": 1, "pad_mode": None},
]


@pytest.mark.parametrize("args", GOOD_INPUTS)
def test_valid_args(args):
    p = Patch(**args)
    for arg, value in args.items():
        if arg in {"patch_size", "overlap"}:
            assert getattr(p.config, arg)[0] == value

    for arg, value in args.items():
        if arg in {"patch_size", "overlap"}:
            args[arg] = (value, value, value)

    p = Patch(**args)
    for arg, value in args.items():
        assert getattr(p.config, arg) == value


@pytest.mark.parametrize("args", BAD_INPUTS)
def test_bad_args(args):
    with pytest.raises((ValidationError, ValueError)):
        Patch(**args)


def test_num_samples_per_image():
    img = torch.randn(2, 5, 7, 3)

    data_point = DataPoint(
        image=tio.ScalarImage(tensor=img),
        label=1,
        participant="sub-000",
        session="ses-000",
    )

    patch = Patch(patch_size=3, overlap=0)
    assert patch.num_samples_per_image(data_point) == len(
        list(
            SlidingWindowSplitter(patch_size=3, overlap=0)(
                data_point.image.tensor.unsqueeze(0)
            )
        )
    )

    patch = Patch(patch_size=3, overlap=0.5, pad_mode=None)
    assert patch.num_samples_per_image(data_point) == len(
        list(
            SlidingWindowSplitter(patch_size=3, overlap=0.5, pad_mode=None)(
                data_point.image.tensor.unsqueeze(0)
            )
        )
    )

    patch = Patch(patch_size=(3, 4, 2), overlap=0.5)
    assert patch.num_samples_per_image(data_point) == len(
        list(
            SlidingWindowSplitter(patch_size=(3, 4, 2), overlap=0.5)(
                data_point.image.tensor.unsqueeze(0)
            )
        )
    )


def test_extract_sample():
    patch = Patch(patch_size=(2, 3, 2), overlap=0.7)
    monai_patch = SlidingWindowSplitter(patch_size=(2, 3, 2), overlap=0.7)

    affine = np.diag([3, 2, 1, 1])
    image_tensor = torch.randn(1, 5, 7, 3)
    mask_1 = torch.randint(0, 2, (1, 5, 7, 3))
    label = torch.randint(0, 2, (3, 5, 7, 3))
    data_point = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor, affine=affine),
        label=tio.LabelMap(tensor=label, affine=affine),
        participant="sub-000",
        session="ses-M000",
        image_path="abc.nii.gz",
        mask_1=tio.LabelMap(tensor=mask_1, affine=affine),
    )

    extracted_data_point = patch(data_point, sample_index=5)

    expected, location = list(monai_patch(data_point.image.tensor.unsqueeze(0)))[5]
    assert isinstance(extracted_data_point.image, tio.ScalarImage)
    assert (extracted_data_point.image.tensor == expected.squeeze(0)).all()

    expected, location = list(monai_patch(data_point.label.tensor.unsqueeze(0)))[5]
    assert isinstance(extracted_data_point.label, tio.LabelMap)
    assert (extracted_data_point.label.tensor == expected.squeeze(0)).all()

    expected, location = list(monai_patch(data_point["mask_1"].tensor.unsqueeze(0)))[5]
    assert isinstance(extracted_data_point["mask_1"], tio.LabelMap)
    assert (extracted_data_point["mask_1"].tensor == expected.squeeze(0)).all()

    assert np.isclose(extracted_data_point.image.affine, affine).all()
    assert np.isclose(extracted_data_point.label.affine, affine).all()

    assert extracted_data_point.participant == "sub-000"
    assert extracted_data_point.session == "ses-M000"
    assert extracted_data_point["image_path"] == "abc.nii.gz"
    assert extracted_data_point["sample_position"] == location

    assert data_point.image.tensor.shape == (1, 5, 7, 3)

    # test transforms history
    transform = tio.Clamp(out_min=0, out_max=10)
    sample = patch(transform(data_point), sample_index=0)
    assert len(sample.get_applied_transforms()) == 1
    assert isinstance(sample.get_applied_transforms()[0], tio.Clamp)

    # other tests
    patch = Patch(patch_size=(2, 3, 2), overlap=(0.5, 0.8, 0))
    monai_patch = SlidingWindowSplitter(patch_size=(2, 3, 2), overlap=(0.5, 0.8, 0))
    extracted_data_point = patch(data_point, sample_index=2)
    expected, location = list(monai_patch(data_point.image.tensor.unsqueeze(0)))[2]
    assert (extracted_data_point.image.tensor == expected.squeeze(0)).all()
    assert extracted_data_point["sample_position"] == location

    patch = Patch(patch_size=(2, 3, 2), overlap=(0.5, 0.8, 0), pad_mode=None)
    monai_patch = SlidingWindowSplitter(
        patch_size=(2, 3, 2), overlap=(0.5, 0.8, 0), pad_mode=None
    )
    extracted_data_point = patch(data_point, sample_index=-1)
    expected, location = list(monai_patch(data_point.image.tensor.unsqueeze(0)))[-1]
    assert (extracted_data_point.image.tensor == expected.squeeze(0)).all()
    assert extracted_data_point["sample_position"] == location

    # errors
    with pytest.raises(IndexError):
        patch(data_point, sample_index=25)

    # generator
    patch = Patch(patch_size=(3, 3, 3), overlap=0, pad_mode=None)
    gen = patch(data_point)
    list_sample_indices = [sample["sample_position"] for sample in gen]
    assert list_sample_indices == [(0, 0, 0), (0, 3, 0)]
