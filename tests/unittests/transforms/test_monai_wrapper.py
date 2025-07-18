import numpy as np
import pytest
import torch
import torchio as tio
from monai.transforms import AsDiscrete

from clinicadl.data.structures import DataPoint
from clinicadl.transforms.monai_wrapper import MonaiTransformWrapper

tensor = torch.tensor([0, 2])
img = tensor.expand((1, 2, 2, 2))

X = DataPoint(
    image=tio.ScalarImage(tensor=img),
    label=tio.LabelMap(tensor=img),
    participant="a",
    session="b",
)
X["array"] = tensor.numpy()
X["tensor"] = tensor
X["numeric"] = 0.2
X["exclude"] = 0.2


def test_monai_wrapper():
    as_discrete = AsDiscrete(threshold=1)
    transform = MonaiTransformWrapper(
        as_discrete, include=["label", "image", "array", "tensor", "numeric"]
    )
    out = transform(X)
    assert (out.label.tensor.unique() == torch.tensor([0, 1])).all()
    assert (out.image.tensor.unique() == torch.tensor([0, 1])).all()
    assert (np.unique(out["array"]) == np.array([0, 1])).all()
    assert (out["tensor"].unique() == torch.tensor([0, 1])).all()
    assert out["numeric"] == 0
    assert out["exclude"] == 0.2

    transform = MonaiTransformWrapper(as_discrete, include=["participant"])
    with pytest.raises(
        TypeError, match="To apply AsDiscrete, 'participant' must be a torchio.Image*"
    ):
        transform(X)
