import numpy as np
import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.data.structures import DataPoint
from clinicadl.transforms import TransformsHandler
from clinicadl.transforms.config import (
    ActivationsConfig,
    PadConfig,
    RescaleIntensityConfig,
    ResizeConfig,
    ToCanonicalConfig,
)
from clinicadl.transforms.extraction import Patch


def test_args():
    with pytest.raises(ValidationError):
        TransformsHandler(extraction=tio.ZNormalization())
    with pytest.raises(ValidationError):
        TransformsHandler(image_transforms=["ZNormalization"])


def test_apply_transforms():
    affine = np.diag([3, 2, 1, 1])
    image = tio.ScalarImage(tensor=torch.randn(1, 14, 14, 14), affine=affine)
    label = tio.LabelMap(tensor=torch.ones(1, 14, 14, 14) * 2, affine=affine)
    label.tensor[:, :2, :2, :2] = 0
    data_point = DataPoint(image, label=label, participant="abc", session="0")
    transforms = TransformsHandler(
        extraction=Patch(patch_size=4, overlap=0),
        image_transforms=[
            tio.Crop(1, copy=False),
            RescaleIntensityConfig(),
        ],
        sample_transforms=[
            PadConfig(padding=1),
            ActivationsConfig(softmax=True, include=["image"]),
        ],
        augmentations=[
            tio.Mask(masking_method="label"),
        ],
    )

    out = transforms.apply_image_transforms(data_point)
    assert out is data_point
    assert data_point.image.tensor.min() == 0
    assert data_point.image.tensor.max() == 1
    assert data_point.image.tensor.shape == (1, 12, 12, 12)
    assert data_point.label.tensor.max() == 2
    assert data_point.label.tensor.shape == (1, 12, 12, 12)

    tio_sample = transforms.extract_sample(data_point, 0)
    assert (tio_sample.image.tensor == data_point.image.tensor[:, :4, :4, :4]).all()
    assert (tio_sample.label.tensor == data_point.label.tensor[:, :4, :4, :4]).all()

    out = transforms.apply_sample_transforms(tio_sample)
    assert out is tio_sample
    assert tio_sample.image.tensor.shape == (1, 6, 6, 6)
    assert tio_sample.label.tensor.shape == (1, 6, 6, 6)
    assert (tio_sample.image.tensor == 1).all()

    out = transforms.apply_augmentations(tio_sample)
    assert out is not tio_sample
    assert (out.image.tensor[:, :2, :2, :2] == 0.0).all()
    assert np.isclose(out.image.affine, affine).all()
    assert np.isclose(out.label.affine, affine).all()


def test_str():
    transforms = TransformsHandler(
        image_transforms=[RescaleIntensityConfig()],
        augmentations=[tio.RescaleIntensity()],
    )
    assert (
        str(transforms)
        == "TransformsHandler configuration for image extraction:\n* image transformation:\n  - RescaleIntensity\n* No sample transformation applied.\n* sample augmentation:\n  - RescaleIntensity\n"
    )


def test_serialization():
    transforms = TransformsHandler(
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

    new_transforms = TransformsHandler.from_dict(d)
    assert isinstance(new_transforms, TransformsHandler)
    assert isinstance(
        new_transforms.config.image_transforms.values[0].value, ResizeConfig
    )
    assert isinstance(
        new_transforms.config.sample_transforms.values[0].value, tio.Resize
    )
    assert isinstance(new_transforms.extraction, Patch)
    assert new_transforms.extraction.config.patch_size == (3, 3, 3)
