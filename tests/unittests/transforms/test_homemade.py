from unittest.mock import patch

import numpy as np
import pytest
import torch
import torchio as tio

from clinicadl.data.structures import DataPoint
from clinicadl.transforms import Format, MergeFields
from clinicadl.utils.numerics import merge_numerics


@pytest.fixture
def datapoint():
    return DataPoint(
        image=tio.ScalarImage(tensor=torch.randn(1, 1, 1, 1)),
        participant="sub-000",
        session="ses-000",
        float=1,
        array_1=np.array([1, 2]),
        array_2=np.array([[[1], [0]]]),
        tensor_1=torch.tensor([3, 4]),
        tensor_2=torch.tensor([[[1], [0]]]),
        mask_1=tio.ScalarImage(tensor=torch.ones(1, 1, 1, 1)),
        mask_2=tio.LabelMap(tensor=torch.ones(1, 1, 1, 1) * 2),
    )


class TestFormat:
    def test_1(self, datapoint):
        transform = Format(
            dtype="float64", squeeze=True, include=["array_2"], copy=False
        )
        out = transform(datapoint)
        np.testing.assert_allclose(out["array_2"], np.array([1, 0], dtype="float64"))
        assert out["tensor_2"].dtype == torch.int64
        assert out is datapoint

    def test_2(self, datapoint):
        transform = Format(
            dtype=torch.float16,
            squeeze=2,
            exclude=[
                "image",
                "participant",
                "session",
                "array_1",
                "tensor_1",
                "mask_1",
                "mask_2",
                "float",
            ],
            copy=True,
        )
        out = transform(datapoint)
        np.testing.assert_allclose(out["array_2"], np.array([[1, 0]], dtype=np.float16))
        torch.testing.assert_close(
            out["tensor_2"], torch.tensor([[1, 0]], dtype=torch.float16)
        )
        assert out is not datapoint

    def test_3(self, datapoint):
        transform = Format(
            squeeze=[0, 2], include=["tensor_2", "array_2"], dtype="float16"
        )
        out = transform(datapoint)
        np.testing.assert_allclose(out["array_2"], np.array([1, 0], dtype=np.float16))
        torch.testing.assert_close(
            out["tensor_2"], torch.tensor([1, 0], dtype=torch.float16)
        )

    def test_4(self, datapoint):
        transform = Format(unsqueeze=1, include=["array_1"])
        out = transform(datapoint)
        np.testing.assert_allclose(out["array_1"], np.array([[1], [2]], dtype=np.int64))

    def test_5(self, datapoint):
        transform = Format(unsqueeze=0, include=["float"], dtype="float64")
        out = transform(datapoint)
        torch.testing.assert_close(out["float"], torch.tensor([1], dtype=torch.float64))


@patch("clinicadl.transforms.homemade.merge_numerics", wraps=merge_numerics)
@pytest.mark.parametrize(
    "keys,output,copy",
    [
        (("tensor_1", "array_1"), torch.tensor([[3, 4], [1, 2]]), False),
        (
            ("mask_1", "mask_2"),
            tio.ScalarImage(tensor=torch.tensor([[[[1.0]]], [[[2.0]]]])),
            True,
        ),
    ],
)
def test_MergeFields(merge_numerics_mock, datapoint, keys, output, copy):
    merger = MergeFields(*keys, output_key="output_key", copy=copy)
    out = merger(datapoint)
    assert isinstance(out["output_key"], type(output))
    merge_numerics_mock.assert_called_once()
    args = merge_numerics_mock.call_args_list[0][0][0]  # first call, args, first arg
    assert len(args) == 2

    try:
        torch.testing.assert_close(out["output_key"], output)
        torch.testing.assert_close(args[0], datapoint[keys[0]])
        torch.testing.assert_close(args[1], datapoint[keys[1]])
    except TypeError:
        torch.testing.assert_close(out["output_key"].tensor, output.tensor)
        torch.testing.assert_close(args[0].tensor, datapoint[keys[0]].tensor)
        torch.testing.assert_close(args[1].tensor, datapoint[keys[1]].tensor)

    assert (out is datapoint) == (not copy)
