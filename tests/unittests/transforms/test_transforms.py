from copy import deepcopy

import numpy as np
import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.data.structures import DataPoint
from clinicadl.transforms import Patch, Transforms, get_transform_config


def test_args():
    with pytest.raises(ValidationError):
        Transforms(extraction=tio.ZNormalization())
    with pytest.raises(ValidationError):
        Transforms(image_transforms=["ZNormalization"])


def test_check_transforms():
    transforms = Transforms(
        image_transforms=[get_transform_config("ZNormalization")],
        sample_transforms=[tio.Resize((16, 16, 16))],
        image_augmentations=[get_transform_config("RandomBlur")],
        sample_augmentations=[tio.RandomAffine()],
    )
    assert [type(t) for t in transforms._image_transforms_processed] == [
        tio.ZNormalization,
        tio.Resize,
    ]
    assert [type(t) for t in transforms._image_augmentations_processed] == [
        tio.RandomBlur,
        tio.RandomAffine,
    ]
    assert transforms.sample_transforms == []
    assert transforms.sample_augmentations == []


def test_get_transforms():
    image = torch.randn(1, 14, 14, 14)
    label = torch.randint(0, 3, (1, 14, 14, 14))
    mask_1 = torch.zeros(1, 14, 14, 14)
    mask_1[:, 2:12, 2:12, 2:12] = 1
    data_point = DataPoint(image, label, mask_1=mask_1)
    transforms = Transforms(
        extraction=Patch(patch_size=4, stride=4),
        image_transforms=[
            tio.Crop(1),
            get_transform_config("RescaleIntensity", padding=1),
        ],
        sample_transforms=[get_transform_config("Pad", padding=1)],
        image_augmentations=[],
        sample_augmentations=[tio.Mask(masking_method="mask_1")],
    )
    (
        image_transforms,
        sample_transforms,
        image_augmentations,
        sample_augmentations,
    ) = transforms.get_transforms()

    data_point = image_transforms(data_point)
    assert data_point.image.tensor.min() == 0
    assert data_point.image.tensor.max() == 1
    assert data_point.image.tensor.shape == (1, 12, 12, 12)
    assert data_point.label.tensor.max() == 2
    assert data_point.label.tensor.shape == (1, 12, 12, 12)
    assert data_point.mask_1.tensor.shape == (1, 12, 12, 12)

    old_data_point = deepcopy(data_point)
    data_point = image_augmentations(data_point)
    assert (data_point.image.tensor == old_data_point.image.tensor).all()
    assert (data_point.label.tensor == old_data_point.label.tensor).all()
    assert (data_point.mask_1.tensor == old_data_point.mask_1.tensor).all()

    tio_sample, _ = transforms.extraction.extract_sample(data_point, 0)
    patch_mask = np.zeros((1, 4, 4, 4))
    patch_mask[:, 1:, 1:, 1:] = 1
    patch_mask = torch.from_numpy(patch_mask)
    assert (tio_sample.image.tensor == data_point.image.tensor[:, :4, :4, :4]).all()
    assert (tio_sample.label.tensor == data_point.label.tensor[:, :4, :4, :4]).all()
    assert (tio_sample.mask_1.tensor == patch_mask).all()

    tio_sample = sample_transforms(tio_sample)
    assert tio_sample.image.tensor.shape == (1, 6, 6, 6)
    assert tio_sample.label.tensor.shape == (1, 6, 6, 6)
    assert tio_sample.mask_1.tensor.shape == (1, 6, 6, 6)

    tio_sample = sample_augmentations(tio_sample)
    assert (tio_sample.image.tensor[:, :2, :2, :2] == 0.0).all()
    assert (tio_sample.image.tensor[:, 5:, 5:, 5:] == 0.0).all()


def test_str():
    transforms = Transforms(
        extraction=Patch(patch_size=4, stride=4),
        image_transforms=[tio.Resize(12), tio.RescaleIntensity()],
        sample_transforms=[get_transform_config("Resize", target_shape=3)],
        image_augmentations=[],
        sample_augmentations=[tio.Mask(masking_method=1)],
    )
    str(transforms)
