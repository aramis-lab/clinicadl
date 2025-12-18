import re
from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchio as tio
from pydantic import ValidationError

from clinicadl.data.dataloader import Batch
from clinicadl.data.datatypes import DataType
from clinicadl.data.structures import DataPoint, Sample, Sample2D
from clinicadl.infer import SlicesToImageInferer
from clinicadl.transforms.config import ActivationsConfig


def test_simple_inferer_args():
    inferer = SlicesToImageInferer(
        slice_direction=1,
        batch_size=3,
        postprocessing_on_cpu=True,
    )
    assert inferer.config.postprocessing_on_cpu

    with pytest.raises(ValidationError):
        inferer = SlicesToImageInferer(
            slice_direction=4,
            batch_size=3,
            postprocessing_on_cpu=True,
        )


def test_simple_inferer():
    inferer = SlicesToImageInferer(
        slice_direction=1,
        postprocessing=[ActivationsConfig(softmax=True, include=["output"])],
    )
    sample = Sample(
        image=tio.ScalarImage(tensor=torch.randn(2, 5, 5, 5)),
        participant="abc",
        session="abc",
        image_path="abc.nii.gz",
        datatype=DataType(pattern="abc", key="abc"),
    )
    network = nn.Conv2d(2, 4, 3)

    with torch.no_grad():
        out = inferer(
            sample,
            network,
        )
    assert str(out.image_path[0]) == "abc.nii.gz"
    assert isinstance(out["output"], tio.ScalarImage)
    assert out["output"].shape == (4, 3, 5, 3)
    torch.testing.assert_close(out["output"].tensor.sum(0), torch.ones((3, 5, 3)))
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
    assert isinstance(out[0]["output"], tio.ScalarImage)
    assert out[0]["output"].shape == (4, 3, 5, 3)
    torch.testing.assert_close(
        out[0]["output"].tensor.sum(0), torch.ones((3, 5, 3), dtype=torch.half)
    )
    assert out[0] is sample

    # output format
    network.to(dtype=torch.float)

    sample["label"] = tio.LabelMap(
        tensor=torch.randint(0, 2, (4, 3, 5, 3)), affine=np.diag([1.2, 1.1, 1, 1])
    )
    with torch.no_grad():
        out = inferer(
            sample,
            network,
        )
    assert isinstance(out["output"], tio.LabelMap)
    assert out["output"].shape == (4, 3, 5, 3)
    np.testing.assert_allclose(out["output"].affine, np.diag([1.2, 1.1, 1, 1]))

    batch = Batch([sample, deepcopy(sample)])
    with torch.no_grad():
        out = inferer(
            batch,
            network,
        )
    assert str(out[0].image_path[0]) == "abc.nii.gz"
    assert isinstance(out[0]["output"], tio.LabelMap)

    # errors
    sample = Sample2D(
        image=tio.ScalarImage(tensor=torch.randn(2, 5, 1, 5)),
        participant="abc",
        session="abc",
        image_path="abc.nii.gz",
        datatype=DataType(pattern="abc", key="abc"),
        sample_position=0,
        slice_direction=1,
        squeeze=True,
    )
    batch = Batch([sample, deepcopy(sample)])
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "SlicesToImageInferer only accepts 4D images (including 1 channel dimension). Got a batch of images with shape: torch.Size([2, 5, 5])"
        ),
    ):
        with torch.no_grad():
            out = inferer(
                batch,
                network,
            )


def test_from_to_dict():
    inferer = SlicesToImageInferer(
        slice_direction=2,
        batch_size=3,
        postprocessing=[ActivationsConfig(softmax=True, include=["output"])],
        postprocessing_on_cpu=True,
    )
    new_inferer = SlicesToImageInferer.from_dict(inferer.to_dict())
    assert new_inferer.config.slice_direction == 2
    assert new_inferer.config.batch_size == 3
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
    network = nn.Conv2d(2, 4, 3)

    inferer = SlicesToImageInferer(
        slice_direction=1,
        postprocessing=[ActivationsConfig(softmax=True, include=["output"])],
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

    inferer = SlicesToImageInferer(
        slice_direction=1,
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
    inferer = SlicesToImageInferer(
        slice_direction=1,
        postprocessing_on_cpu=True,
    )
    with torch.no_grad():
        out = inferer(
            batch,
            network,
        )
    assert out.device == torch.device("cuda")  # no postprocessing
