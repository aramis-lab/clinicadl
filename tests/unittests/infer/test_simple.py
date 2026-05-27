from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchio as tio

from clinicadl.data.dataloader import Batch
from clinicadl.data.structures import DataPoint, Sample, Sample2D
from clinicadl.infer import SimpleInferer
from clinicadl.io import BidsFileType
from clinicadl.transforms.config import ActivationsConfig

from .utils import NnWrapper


def test_inferer():
    inferer = SimpleInferer(
        postprocessing=[ActivationsConfig(softmax=True, include=["output"])]
    )
    sample = Sample(
        image=tio.ScalarImage(
            tensor=torch.randn(2, 3, 3, 3), affine=np.diag([1.2, 1.1, 1, 1])
        ),
        participant="abc",
        session="abc",
        image_path="abc.nii.gz",
        file_type=BidsFileType(data_type="abc", suffix="abc"),
    )
    network = nn.Sequential(nn.Flatten(start_dim=-4), nn.Linear(2 * 3**3, 2))

    with torch.no_grad():
        out = inferer(
            sample,
            network,
        )
    assert str(out.image_path[0]) == "abc.nii.gz"
    assert isinstance(out["output"], torch.Tensor)
    assert out["output"].size() == torch.Size([2])
    torch.testing.assert_close(out["output"].sum(), torch.tensor(1.0))
    assert out is sample

    # batch
    network.to(dtype=torch.half)
    batch = Batch([sample, deepcopy(sample)])
    with torch.no_grad():
        out = inferer(
            batch,
            network,
            input_dtype=torch.half,
        )
    assert str(out[0].image_path[0]) == "abc.nii.gz"
    assert out[0]["output"].size() == torch.Size([2])
    torch.testing.assert_close(
        out[0]["output"].sum(), torch.tensor(1.0, dtype=torch.half)
    )
    assert out[0] is sample

    # output format and name
    network = nn.Identity()
    inferer = SimpleInferer(
        output_name="my_output",
        postprocessing=[ActivationsConfig(softmax=True, include=["my_output"])],
        output_type="image",
    )

    with torch.no_grad():
        out = inferer(
            sample,
            network,
        )
    assert isinstance(out["my_output"], tio.ScalarImage)
    torch.testing.assert_close(out["my_output"].tensor.sum(), torch.tensor(27.0))
    np.testing.assert_allclose(out["my_output"].affine, np.diag([1.2, 1.1, 1, 1]))

    batch = Batch([sample, deepcopy(sample)])
    with torch.no_grad():
        out = inferer(
            batch,
            network,
        )
    assert isinstance(out[0]["my_output"], tio.ScalarImage)
    torch.testing.assert_close(out[0]["my_output"].tensor.sum(), torch.tensor(27.0))
    np.testing.assert_allclose(out[0]["my_output"].affine, np.diag([1.2, 1.1, 1, 1]))

    # mask
    inferer = SimpleInferer(
        output_type="mask",
    )
    with torch.no_grad():
        out = inferer(
            sample,
            network,
        )
    assert isinstance(out["output"], tio.LabelMap)
    np.testing.assert_allclose(out["output"].affine, np.diag([1.2, 1.1, 1, 1]))

    # kwargs
    network = NnWrapper(network)
    inferer = SimpleInferer()
    with torch.no_grad():
        out = inferer(
            sample,
            network,
        )
        out_ = inferer(
            deepcopy(sample),
            network,
            offset=1,
        )
    torch.testing.assert_close(out["output"] + 1, out_["output"])

    # 2D
    sample = Sample2D(
        image=tio.ScalarImage(tensor=torch.randn(2, 5, 1, 5)),
        participant="abc",
        session="abc",
        image_path="abc.nii.gz",
        file_type=BidsFileType(data_type="abc", suffix="abc"),
        sample_position=0,
        slice_direction=1,
        squeeze=True,
    )
    network = nn.Conv2d(2, 4, 3)

    inferer = SimpleInferer(output_type="image")
    with torch.no_grad():
        out = inferer(
            sample,
            network,
        )
    assert out["output"].shape == (4, 3, 1, 3)

    inferer = SimpleInferer(output_type="mask")
    batch = Batch([sample, deepcopy(sample)])
    with torch.no_grad():
        out = inferer(
            batch,
            network,
        )
    assert out[0]["output"].shape == (4, 3, 1, 3)


def test_from_to_dict():
    inferer = SimpleInferer(
        postprocessing=[ActivationsConfig(softmax=True, include=["output"])],
        postprocessing_on_cpu=True,
    )
    new_inferer = SimpleInferer.from_dict(inferer.to_dict())
    assert new_inferer.config.postprocessing_on_cpu
    assert isinstance(
        new_inferer.config.postprocessing.config.transforms.values[0].value,
        ActivationsConfig,
    )


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

    batch.to("cuda")
    inferer = SimpleInferer(
        postprocessing_on_cpu=True,
    )
    with torch.no_grad():
        out = inferer(
            batch,
            network,
        )
    assert out.device == torch.device("cuda")  # no postprocessing
