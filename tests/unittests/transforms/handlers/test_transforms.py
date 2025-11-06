import numpy as np
import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.data.structures import DataPoint
from clinicadl.transforms import Transforms
from clinicadl.transforms.config import (
    ActivationsConfig,
    PadConfig,
    RescaleIntensityConfig,
    ResizeConfig,
    ToCanonicalConfig,
    ZNormalizationConfig,
)
from clinicadl.transforms.extraction import Patch


def test_args():
    with pytest.raises(ValidationError):
        Transforms(extraction=tio.ZNormalization())
    with pytest.raises(ValidationError):
        Transforms(image_transforms=["ZNormalization"])


def test_check_transforms():
    transforms = Transforms(
        image_transforms=[ZNormalizationConfig()],
        sample_transforms=[tio.Resize((16, 16, 16))],
        augmentations=[tio.RandomAffine()],
    )
    assert [type(t) for t in transforms.image_transforms] == [
        tio.ZNormalization,
        tio.Resize,
    ]
    assert transforms.sample_transforms.transforms == []


def test_apply_transforms():
    affine = np.diag([3, 2, 1, 1])
    image = tio.ScalarImage(tensor=torch.randn(1, 14, 14, 14), affine=affine)
    label = tio.LabelMap(tensor=torch.randint(0, 3, (1, 14, 14, 14)), affine=affine)
    mask_1 = torch.zeros(1, 14, 14, 14)
    mask_1[:, 2:12, 2:12, 2:12] = 1
    mask_1 = tio.LabelMap(tensor=mask_1, affine=affine)
    data_point = DataPoint(image, label, mask_1=mask_1, participant="abc", session="0")
    transforms = Transforms(
        extraction=Patch(patch_size=4, overlap=0),
        image_transforms=[
            tio.Crop(1),
            RescaleIntensityConfig(),
        ],
        sample_transforms=[
            PadConfig(padding=1),
            ActivationsConfig(softmax=True, include=["image"]),
        ],
        augmentations=[
            tio.Mask(masking_method="mask_1"),
        ],
    )

    data_point = transforms.apply_image_transforms(data_point)
    assert data_point.image.tensor.min() == 0
    assert data_point.image.tensor.max() == 1
    assert data_point.image.tensor.shape == (1, 12, 12, 12)
    assert data_point.label.tensor.max() == 2
    assert data_point.label.tensor.shape == (1, 12, 12, 12)
    assert data_point.mask_1.tensor.shape == (1, 12, 12, 12)

    tio_sample = transforms.extract_sample(data_point, 0)
    patch_mask = np.zeros((1, 4, 4, 4))
    patch_mask[:, 1:, 1:, 1:] = 1
    patch_mask = torch.from_numpy(patch_mask)
    assert (tio_sample.image.tensor == data_point.image.tensor[:, :4, :4, :4]).all()
    assert (tio_sample.label.tensor == data_point.label.tensor[:, :4, :4, :4]).all()
    assert (tio_sample.mask_1.tensor == patch_mask).all()

    tio_sample = transforms.apply_sample_transforms(tio_sample)
    assert tio_sample.image.tensor.shape == (1, 6, 6, 6)
    assert tio_sample.label.tensor.shape == (1, 6, 6, 6)
    assert tio_sample.mask_1.tensor.shape == (1, 6, 6, 6)
    assert (tio_sample.image.tensor == 1).all()

    tio_sample = transforms.apply_augmentations(tio_sample)
    assert (tio_sample.image.tensor[:, :2, :2, :2] == 0.0).all()
    assert (tio_sample.image.tensor[:, 5:, 5:, 5:] == 0.0).all()
    assert np.isclose(tio_sample.image.affine, affine).all()
    assert np.isclose(tio_sample.label.affine, affine).all()
    assert np.isclose(tio_sample.mask_1.affine, affine).all()


def test_str():
    transforms = Transforms(
        image_transforms=[RescaleIntensityConfig()],
        augmentations=[tio.RescaleIntensity()],
    )
    assert (
        str(transforms)
        == "Transforms configuration for image extraction:\n* image transformation:\n  - RescaleIntensity\n* No sample transformation applied.\n* sample augmentation:\n  - RescaleIntensity\n"
    )


def test_serialization():
    transforms = Transforms(
        extraction=Patch(patch_size=3),
        image_transforms=[
            ResizeConfig(target_shape=3),
        ],
        sample_transforms=[
            tio.Resize(target_shape=3),
        ],
        augmentations=[ToCanonicalConfig()],
    )
    d = transforms.to_dict()

    new_transforms = Transforms.from_dict(d)
    assert isinstance(new_transforms, Transforms)
    assert isinstance(
        new_transforms.config.image_transforms.values[0].value, ResizeConfig
    )
    assert isinstance(
        new_transforms.config.sample_transforms.values[0].value, tio.Resize
    )
    assert isinstance(new_transforms.extraction, Patch)
    assert new_transforms.extraction.config.patch_size == (3, 3, 3)
