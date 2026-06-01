import re
from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchio as tio
from pydantic import ValidationError

from clinicadl.data.dataloader import Batch
from clinicadl.data.structures import DataPoint, Sample, Sample2D
from clinicadl.infer import PatchesToImageInferer
from clinicadl.io.bids import BidsFileType
from clinicadl.transforms.config import ActivationsConfig

from .utils import NnWrapper

BAD_ARGS = [
    {"patch_size": (0, 1, 1)},
    {"overlap": 1.1},
    {"avg_mode": "abc"},
    {"sigma_scale": 0},
    {"batch_size": 0},
]


@pytest.mark.parametrize("args", BAD_ARGS)
def test_bad_args(args):
    with pytest.raises(ValidationError):
        if "patch_size" not in args:
            PatchesToImageInferer(patch_size=1, **args)
        else:
            PatchesToImageInferer(**args)


def test_args():
    inferer = PatchesToImageInferer(
        patch_size=10,
        overlap=0.25,
        avg_mode="gaussian",
        sigma_scale=0.5,
        batch_size=1,
    )
    assert inferer._sliding_window.roi_size == (10, 10, 10)
    assert inferer._sliding_window.overlap == (0.25, 0.25, 0.25)
    assert inferer._sliding_window.mode == "gaussian"
    assert inferer._sliding_window.sigma_scale == 0.5
    assert inferer._sliding_window.sw_batch_size == 1


def test_inferer():
    inferer = PatchesToImageInferer(
        patch_size=(3, 2, 2),
        overlap=1 / 3,
    )
    sample = Sample(
        image=tio.ScalarImage(
            tensor=torch.randn(1, 4, 4, 2), affine=np.diag([1.2, 1.1, 1, 1])
        ),
        participant="abc",
        session="abc",
        image_path="abc.nii.gz",
        file_type=BidsFileType(data_type="abc", suffix="abc"),
    )
    network = nn.Identity()

    slices = [
        (slice(0, 3), slice(0, 2), ...),
        (slice(1, 4), slice(0, 2), ...),
        (slice(0, 3), slice(1, 3), ...),
        (slice(1, 4), slice(1, 3), ...),
        (slice(0, 3), slice(2, 4), ...),
        (slice(1, 4), slice(2, 4), ...),
    ]
    expected_output = torch.zeros_like(sample.image.tensor)
    cnt = torch.zeros_like(sample.image.tensor)
    for slice_ in slices:
        expected_output[slice_] += sample.image.tensor[slice_]
        cnt[slice_] += 1
    expected_output /= cnt

    with torch.no_grad():
        out = inferer(
            sample,
            network,
        )
    assert str(out.image_path[0]) == "abc.nii.gz"
    assert isinstance(out["output"], torch.Tensor)
    torch.testing.assert_close(out["output"], expected_output)
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
    torch.testing.assert_close(out[0]["output"], expected_output.to(dtype=torch.half))
    assert out[0] is sample

    # output format and name
    network.to(dtype=torch.float)
    inferer = PatchesToImageInferer(
        patch_size=(3, 2, 2),
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
    torch.testing.assert_close(out["my_output"].tensor.sum(0), torch.ones((4, 4, 2)))
    np.testing.assert_allclose(out["my_output"].affine, np.diag([1.2, 1.1, 1, 1]))

    batch = Batch([sample, deepcopy(sample)])
    with torch.no_grad():
        out = inferer(
            batch,
            network,
        )
    assert isinstance(out[0]["my_output"], tio.ScalarImage)
    torch.testing.assert_close(out[0]["my_output"].tensor.sum(0), torch.ones((4, 4, 2)))
    np.testing.assert_allclose(out[0]["my_output"].affine, np.diag([1.2, 1.1, 1, 1]))

    # mask
    inferer = PatchesToImageInferer(
        patch_size=(3, 2, 2),
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
    inferer = PatchesToImageInferer(patch_size=(3, 2, 2))
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

    # errors
    inferer = PatchesToImageInferer(
        patch_size=(5, 2, 2),
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "'patch_size' is bigger than the image. Got an image of spatial shape torch.Size([4, 4, 2]) but patch_size=(5, 2, 2)"
        ),
    ):
        with torch.no_grad():
            inferer(
                batch,
                network,
            )

    inferer = PatchesToImageInferer(
        patch_size=(3, 2, 2),
    )
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
    batch = Batch([sample, deepcopy(sample)])
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "PatchesToImageInferer only accepts 4D images (including 1 channel dimension). Got a batch of images with shape: torch.Size([2, 5, 5])"
        ),
    ):
        with torch.no_grad():
            inferer(
                batch,
                network,
            )


def test_from_to_dict():
    inferer = PatchesToImageInferer(
        patch_size=10,
        overlap=0.25,
        avg_mode="gaussian",
        sigma_scale=0.5,
        batch_size=3,
        postprocessing=[ActivationsConfig(softmax=True, include=["output"])],
        postprocessing_on_cpu=True,
    )
    new_inferer = PatchesToImageInferer.from_dict(inferer.to_dict())
    assert new_inferer.config.patch_size == (10, 10, 10)
    assert new_inferer.config.avg_mode == "gaussian"
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
    network = nn.Identity()

    inferer = PatchesToImageInferer(
        patch_size=(3, 2, 2),
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

    inferer = PatchesToImageInferer(
        patch_size=(3, 2, 2),
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
    inferer = PatchesToImageInferer(
        patch_size=(3, 2, 2),
        postprocessing_on_cpu=True,
    )
    with torch.no_grad():
        out = inferer(
            batch,
            network,
        )
    assert out.device == torch.device("cuda")  # no postprocessing
