import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.transforms.extraction import Patch
from clinicadl.transforms.transforms import Transforms


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
    tensor = torch.randn(1, 13, 13, 13)
    transforms = Transforms(
        extraction=Patch(patch_size=4, stride=4),
        image_transforms=[tio.Resize(12), tio.RescaleIntensity()],
        sample_transforms=[tio.Resize(3)],
        image_augmentations=[],
        sample_augmentations=[tio.Mask(masking_method=1)],
    )
    (
        image_transforms,
        sample_transforms,
        image_augmentations,
        sample_augmentations,
    ) = transforms.get_transforms()

    tensor = image_transforms(tensor)
    assert tensor.min() == 0
    assert tensor.max() == 1
    assert tensor.shape == (1, 12, 12, 12)
    assert (image_augmentations(tensor) == tensor).all()
    patch = tensor[:, :4, :4, :4].clone()
    patch = sample_transforms(patch)
    assert patch.shape == (1, 3, 3, 3)
    patch = sample_augmentations(patch)
    assert (patch != 0).sum() == 1


def test_str():
    transforms = Transforms(
        extraction=Patch(patch_size=4, stride=4),
        image_transforms=[tio.Resize(12), tio.RescaleIntensity()],
        sample_transforms=[tio.Resize(3)],
        image_augmentations=[],
        sample_augmentations=[tio.Mask(masking_method=1)],
    )
    str(transforms)
