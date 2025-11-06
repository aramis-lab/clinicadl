from collections import OrderedDict

import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.data.structures import DataPoint
from clinicadl.transforms.config import AsDiscreteConfig
from clinicadl.transforms.handlers import Postprocessing
from clinicadl.transforms.monai_wrapper import MonaiTransformWrapper


def test_args():
    with pytest.raises(ValidationError):
        Postprocessing(transforms=["AsDiscrete"])


def test_check_transforms():
    transforms = Postprocessing(
        transforms=[AsDiscreteConfig(threshold=1), tio.RescaleIntensity()],
    )
    assert [type(t) for t in transforms.transforms] == [
        MonaiTransformWrapper,
        tio.RescaleIntensity,
    ]


def test_apply():
    data_point = DataPoint(
        tio.ScalarImage(tensor=torch.randint(0, 3, (1, 2, 2, 2))),
        label=torch.tensor([0, 2]),
        participant="abc",
        session="0",
    )
    transforms = Postprocessing(
        transforms=[
            AsDiscreteConfig(threshold=1, include=["label"]),
            tio.RescaleIntensity(),
        ],
    )

    data_point = transforms.apply(data_point)
    assert data_point.image.tensor.min() == 0
    assert data_point.image.tensor.max() == 1
    torch.testing.assert_close(data_point.label, torch.tensor([0.0, 1.0]))

    # batch
    data_point.image = (tio.ScalarImage(tensor=torch.randint(0, 3, (1, 2, 2, 2))),)
    batch = [data_point, data_point]
    batch = transforms.batch_apply(batch)
    assert batch[0].image.tensor.max() == 1
    assert batch[1].image.tensor.max() == 1


def test_str():
    transforms = Postprocessing(
        transforms=[tio.RescaleIntensity(), AsDiscreteConfig(threshold=1)]
    )
    assert str(transforms) == "Postprocessing:\n  - RescaleIntensity\n  - AsDiscrete\n"
    transforms = Postprocessing(transforms=[])
    assert str(transforms) == "Postprocessing:\nNo transform applied.\n"


def test_serialization():
    transforms = Postprocessing(
        transforms=[
            AsDiscreteConfig(threshold=1),
            tio.Resample(),
        ],
    )
    d = transforms.to_dict()

    new_transforms = Postprocessing.from_dict(d)
    assert isinstance(new_transforms, Postprocessing)
    assert isinstance(
        new_transforms.config.transforms.values[0].value, AsDiscreteConfig
    )
    assert isinstance(new_transforms.config.transforms.values[1].value, tio.Resample)
    assert new_transforms.config.transforms.values[0].value.threshold == 1
