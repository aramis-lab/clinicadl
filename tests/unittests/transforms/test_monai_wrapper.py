import re

import pytest
import torch
import torchio as tio
from monai.transforms import Activations

from clinicadl.data.structures import DataPoint
from clinicadl.transforms.monai_wrapper import MonaiTransformWrapper

tensor = torch.tensor([[0, 2]])
img = tensor.expand((1, 2, 2, 2))

X = DataPoint(
    image=tio.ScalarImage(tensor=img),
    label=0.2,
    mask=tio.LabelMap(tensor=img),
    participant="a",
    session="b",
)
X["array"] = tensor.numpy()
X["tensor"] = tensor
X["exclude"] = 0.2


def test_monai_wrapper():
    as_discrete = Activations(softmax=True)

    transform = MonaiTransformWrapper(as_discrete)
    out = transform(X)
    assert (out.mask.tensor == 1).all()
    assert (out.image.tensor == 1).all()
    assert not (out["array"] == 1).all()
    assert not (out["tensor"] == 1).all()
    assert out.label == 0.2

    transform = MonaiTransformWrapper(
        as_discrete, include=["mask", "image", "array", "tensor", "label"]
    )
    out = transform(X)
    assert (out.mask.tensor == 1).all()
    assert (out.image.tensor == 1).all()
    assert (out["array"] == 1).all()
    assert (out["tensor"] == 1).all()
    assert out["label"] == 1
    assert out["exclude"] == 0.2

    transform = MonaiTransformWrapper(as_discrete, exclude=["mask"])
    out = transform(X)
    assert not (out.mask.tensor == 1).all()
    assert (out.image.tensor == 1).all()
    assert not (out["array"] == 1).all()
    assert not (out["tensor"] == 1).all()
    assert out["label"] == 0.2

    with pytest.raises(
        ValueError,
        match="You cannot pass both 'include' and 'exclude'.",
    ):
        MonaiTransformWrapper(as_discrete, include=["image"], exclude=["label"])

    transform = MonaiTransformWrapper(as_discrete, include=["participant"])
    with pytest.raises(
        TypeError,
        match="To apply 'Activations', 'participant' must be a torchio.Image*",
    ):
        transform(X)


def test_repr():
    as_discrete = Activations(softmax=True)
    transform = MonaiTransformWrapper(
        as_discrete, include=["label", "image", "array", "tensor", "numeric"]
    )
    pattern = r"MonaiTransformWrapper\(transform=<monai\.transforms\.post\.array\.Activations object at .*?>, include=\['label', 'image', 'array', 'tensor', 'numeric'\]\)"
    assert re.fullmatch(pattern, repr(transform))
