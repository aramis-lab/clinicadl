from copy import deepcopy

import numpy as np
import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.transforms.extraction import Patch
from clinicadl.transforms.transforms import Transforms
from clinicadl.transforms.utils import get_tio_image


def test_args():
    with pytest.raises(ValidationError):
        Transforms(extraction=tio.ZNormalization())
    with pytest.raises(ValidationError):
        Transforms(image_transforms=["ZNormalization"])


def test_check_transforms():
    transforms = Transforms(
        image_transforms=[tio.ZNormalization()],
        sample_transforms=[tio.Resize((16, 16, 16))],
        image_augmentations=[tio.RandomBlur()],
        sample_augmentations=[tio.RandomAffine()],
    )
    assert [type(t) for t in transforms.image_transforms] == [
        tio.ZNormalization,
        tio.Resize,
    ]
    assert [type(t) for t in transforms.image_augmentations] == [
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
    tio_image = get_tio_image(image, label, mask_1=mask_1)
    transforms = Transforms(
        extraction=Patch(patch_size=4, stride=4),
        image_transforms=[tio.Crop(1), tio.RescaleIntensity()],
        sample_transforms=[tio.Pad(1)],
        image_augmentations=[],
        sample_augmentations=[tio.Mask(masking_method="mask_1")],
    )
    (
        image_transforms,
        sample_transforms,
        image_augmentations,
        sample_augmentations,
    ) = transforms.get_transforms()

    tio_image = image_transforms(tio_image)
    assert tio_image.image.tensor.min() == 0
    assert tio_image.image.tensor.max() == 1
    assert tio_image.image.tensor.shape == (1, 12, 12, 12)
    assert tio_image.label.tensor.max() == 2
    assert tio_image.label.tensor.shape == (1, 12, 12, 12)
    assert tio_image.mask_1.tensor.shape == (1, 12, 12, 12)

    old_tio_image = deepcopy(tio_image)
    tio_image = image_augmentations(tio_image)
    assert (tio_image.image.tensor == old_tio_image.image.tensor).all()
    assert (tio_image.label.tensor == old_tio_image.label.tensor).all()
    assert (tio_image.mask_1.tensor == old_tio_image.mask_1.tensor).all()

    tio_sample = transforms.extraction.extract_tio_sample(tio_image, 0)
    patch_mask = np.zeros((1, 4, 4, 4))
    patch_mask[:, 1:, 1:, 1:] = 1
    patch_mask = torch.from_numpy(patch_mask)
    assert (tio_sample.sample.tensor == tio_image.image.tensor[:, :4, :4, :4]).all()
    assert (tio_sample.label.tensor == tio_image.label.tensor[:, :4, :4, :4]).all()
    assert (tio_sample.mask_1.tensor == patch_mask).all()

    tio_sample = sample_transforms(tio_sample)
    assert tio_sample.sample.tensor.shape == (1, 6, 6, 6)
    assert tio_sample.label.tensor.shape == (1, 6, 6, 6)
    assert tio_sample.mask_1.tensor.shape == (1, 6, 6, 6)

    tio_sample = sample_augmentations(tio_sample)
    assert (tio_sample.sample.tensor[:, :2, :2, :2] == 0.0).all()
    assert (tio_sample.sample.tensor[:, 5:, 5:, 5:] == 0.0).all()


def test_str():
    transforms = Transforms(
        extraction=Patch(patch_size=4, stride=4),
        image_transforms=[tio.Resize(12), tio.RescaleIntensity()],
        sample_transforms=[tio.Resize(3)],
        image_augmentations=[],
        sample_augmentations=[tio.Mask(masking_method=1)],
    )
    str(transforms)
