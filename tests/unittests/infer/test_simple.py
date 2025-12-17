from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchio as tio

from clinicadl.data.dataloader import Batch
from clinicadl.data.datatypes import DataType
from clinicadl.data.structures import DataPoint, Sample
from clinicadl.infer import SimpleInferer
from clinicadl.transforms.config import ActivationsConfig


def test_simple_inferer():
    inferer = SimpleInferer(
        postprocessing=[ActivationsConfig(softmax=True, include=["output"])]
    )
    sample = Sample(
        image=tio.ScalarImage(tensor=torch.randn(2, 3, 3, 3)),
        participant="abc",
        session="abc",
        image_path="abc.nii.gz",
        datatype=DataType(pattern="abc", key="abc"),
    )
    network = nn.Sequential(nn.Flatten(start_dim=-4), nn.Linear(2 * 3**3, 2))

    with torch.no_grad():
        out = inferer(
            sample,
            network,
        )
    assert str(out.image_path[0]) == "abc.nii.gz"
    assert out["output"].shape == (2,)
    torch.testing.assert_close(out["output"].sum(), torch.tensor(1.0))
    assert out is sample

    batch = Batch([sample, deepcopy(sample)])
    with torch.no_grad():
        out = inferer(
            batch,
            network,
        )
    assert str(out[0].image_path[0]) == "abc.nii.gz"
    assert out[0]["output"].shape == (2,)
    torch.testing.assert_close(out[0]["output"].sum(), torch.tensor(1.0))
    assert out[0] is sample

    # output format
    network = nn.Identity()

    sample["label"] = tio.LabelMap(
        tensor=torch.randint(0, 2, (2, 3, 3, 3)), affine=np.diag([1.2, 1.1, 1, 1])
    )
    with torch.no_grad():
        out = inferer(
            sample,
            network,
        )
    assert isinstance(out["output"], tio.LabelMap)
    assert out["output"].shape == (2, 3, 3, 3)
    torch.testing.assert_close(out["output"].tensor.sum(), torch.tensor(27.0))

    batch = Batch([sample, deepcopy(sample)])
    with torch.no_grad():
        out = inferer(
            batch,
            network,
        )
    assert str(out[0].image_path[0]) == "abc.nii.gz"
    assert out[0]["output"].shape == (2, 3, 3, 3)

    sample["label"] = None
    with torch.no_grad():
        out = inferer(
            sample,
            network,
        )
    assert isinstance(out["output"], tio.ScalarImage)
    assert out["output"].shape == (2, 3, 3, 3)


@pytest.mark.gpu
def test_gpu():
    dp = DataPoint(
        image=tio.ScalarImage(tensor=torch.randn(2, 3, 3, 3)),
        participant="abc",
        session="abc",
    )
    network = nn.Sequential(nn.Flatten(start_dim=-4), nn.Linear(2 * 3**3, 2))
    inferer = SimpleInferer(
        postprocessing=[ActivationsConfig(softmax=True, include=["output"])]
    )

    batch = Batch([dp, deepcopy(dp)])
    network.to("cuda")
    batch.to("cuda")
    with torch.no_grad():
        out = inferer(
            batch,
            network,
        )
    assert out is batch
    assert out.device == torch.device("cuda")

    inferer = SimpleInferer(
        postprocessing=[ActivationsConfig(softmax=True, include=["output"])],
        postprocessing_on_cpu=True,
    )
    with torch.no_grad():
        out = inferer(
            batch,
            network,
        )
    assert out.device == torch.device("cpu")
