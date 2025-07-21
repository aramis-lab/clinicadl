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
    img = torch.randn(2, 5, 7, 3)

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
    extracted_data_point = patch.extract_sample(data_point, sample_index=5)
    assert isinstance(extracted_data_point.image, tio.ScalarImage)
    assert (extracted_data_point.image.tensor == image_tensor[:, :2, 4:7, 1:3]).all()
    assert isinstance(extracted_data_point.label, tio.LabelMap)
    assert (extracted_data_point.label.tensor == label[:, :2, 4:7, 1:3]).all()
    assert isinstance(extracted_data_point["mask_1"], tio.LabelMap)
    assert (extracted_data_point["mask_1"].tensor == mask_1[:, :2, 4:7, 1:3]).all()

    assert np.isclose(extracted_data_point.image.affine, affine).all()
    assert np.isclose(extracted_data_point.label.affine, affine).all()

    assert extracted_data_point.participant == "sub-000"
    assert extracted_data_point.session == "ses-M000"
    assert extracted_data_point.image_path == "abc.nii.gz"
    assert extracted_data_point._sample_index == 5
    assert extracted_data_point.patch_size == (2, 3, 2)
    assert extracted_data_point.patch_stride == (1, 2, 1)
    assert extracted_data_point._sample_index == 5

    assert data_point.image.tensor.shape == (1, 5, 7, 3)

    # test transforms history
    transform = tio.Clamp(out_min=0, out_max=10)
    sample = patch.extract_sample(transform(data_point), sample_index=0)
    assert len(sample.get_applied_transforms()) == 1
    assert isinstance(sample.get_applied_transforms()[0], tio.Clamp)

    # other tests
    data_point = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor),
        label=1,
        participant="sub-000",
        session="ses-000",
    )
    extracted_data_point = patch.extract_sample(data_point, sample_index=5)
    assert extracted_data_point.label == 1

    with pytest.raises(IndexError):
        patch.extract_sample(data_point, sample_index=25)

    patch = Patch(patch_size=2, stride=1)
    extracted_data_point = patch.extract_sample(data_point, sample_index=1)
    assert (
        extracted_data_point.image.tensor == image_tensor[:, :2, :2, 1:3]
    ).all()  # .view starts with the last dimension

    patch = Patch(patch_size=15, stride=1)
    with pytest.raises(IndexError):
        patch.extract_sample(data_point, sample_index=0)
