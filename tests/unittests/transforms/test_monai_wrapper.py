import re

import numpy as np
import pytest
import torch
import torchio as tio
from monai.transforms import Activations

from clinicadl.data.structures import DataPoint
from clinicadl.transforms.monai_wrapper import MonaiTransformWrapper

tensor = torch.tensor([[1, 1], [1, 0]]).float()
img = torch.tensor([[[[1]], [[1]]], [[[1]], [[0]]]]).float()

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
    # only on images
    transform = MonaiTransformWrapper(Activations(softmax=True))
    out = transform(X)
    torch.testing.assert_close(
        out.mask.tensor,
        torch.tensor([[[[0.5000]], [[0.7311]]], [[[0.5000]], [[0.2689]]]]),
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        out.image.tensor,
        torch.tensor([[[[0.5000]], [[0.7311]]], [[[0.5000]], [[0.2689]]]]),
        rtol=1e-3,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        out["array"],
        tensor.numpy(),
        rtol=1e-3,
    )
    torch.testing.assert_close(
        out["tensor"],
        tensor,
        rtol=1e-3,
        atol=1e-3,
    )
    assert out.label == 0.2

    # include
    transform = MonaiTransformWrapper(
        Activations(softmax=True), include=["mask", "image", "array", "tensor", "label"]
    )
    out = transform(X)
    assert isinstance(out["tensor"], torch.Tensor)
    torch.testing.assert_close(
        out["tensor"],
        torch.tensor([[0.5000, 0.7311], [0.5000, 0.2689]]),
        rtol=1e-3,
        atol=1e-3,
    )
    assert isinstance(out["array"], np.ndarray)
    np.testing.assert_allclose(
        out["array"], np.array([[0.5000, 0.7311], [0.5000, 0.2689]]), rtol=1e-3
    )
    assert out.label == 1
    assert out["exclude"] == 0.2

    # exclude
    transform = MonaiTransformWrapper(Activations(softmax=True), exclude=["mask"])
    out = transform(X)
    torch.testing.assert_close(
        out.mask.tensor,
        img,
        rtol=1e-3,
        atol=1e-3,
    )
    assert out.label == 0.2

    # errors
    with pytest.raises(
        ValueError,
        match="You cannot pass both 'include' and 'exclude'.",
    ):
        MonaiTransformWrapper(
            Activations(softmax=True), include=["image"], exclude=["label"]
        )

    transform = MonaiTransformWrapper(
        Activations(softmax=True), include=["participant"]
    )
    with pytest.raises(
        Exception,
        match="An error occurred while transforming the field 'participant'.",
    ):
        transform(X)


def test_repr():
    as_discrete = Activations(softmax=True)
    transform = MonaiTransformWrapper(
        as_discrete, include=["label", "image", "array", "tensor", "numeric"]
    )
    pattern = r"MonaiTransformWrapper\(transform=<monai\.transforms\.post\.array\.Activations object at .*?>, include=\['label', 'image', 'array', 'tensor', 'numeric'\]\)"
    assert re.fullmatch(pattern, repr(transform))
