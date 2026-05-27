from unittest.mock import patch

import numpy as np
import pytest
import torch
import torchio as tio

from clinicadl.utils.numerics import concat_images, merge_numerics


@patch(
    "clinicadl.utils.numerics.concat_images",
    wraps=concat_images,
)
@pytest.mark.parametrize(
    "inputs,expected_output",
    [
        ([1, 2.0], [1, 2.0]),
        ([(1, 2), (3, 4)], [1, 2, 3, 4]),
        ([[1, 2], (3, 4)], [1, 2, 3, 4]),
        ([(1, 2), [3, 4]], [1, 2, 3, 4]),
        ([np.array([1, 2]), np.array([3, 4])], np.array([[1, 2], [3, 4]])),
        ([torch.tensor([1, 2]), torch.tensor([3, 4])], torch.tensor([[1, 2], [3, 4]])),
        ([np.array([1, 2]), torch.tensor([3, 4])], torch.tensor([[1, 2], [3, 4]])),
        (
            [
                tio.ScalarImage(tensor=torch.ones(1, 1, 1, 1)),
                tio.ScalarImage(tensor=torch.ones(1, 1, 1, 1) * 2),
            ],
            tio.ScalarImage(tensor=torch.tensor([[[[1.0]]], [[[2.0]]]])),
        ),
        (
            [np.array([1, 2]), torch.tensor([3, 4])],
            [np.array([1, 2]), torch.tensor([3, 4])],
        ),
        (
            [np.array([1, 2]), np.array([3, 4, 5])],
            None,
        ),
        (
            [torch.tensor([1, 2]), torch.tensor([3, 4, 5])],
            None,
        ),
    ],
)
def test_merge_numerics(spy, inputs, expected_output):
    if expected_output is not None:
        output = merge_numerics(inputs)
    else:
        with pytest.raises(
            RuntimeError,
            match="An error occurred when merging the values, probably because the have different shapes.",
        ):
            merge_numerics(inputs)
        return

    if isinstance(inputs[0], tio.Image):
        spy.assert_called_once()
        torch.testing.assert_close(output.tensor, expected_output.tensor)
    elif isinstance(inputs[0], torch.Tensor):
        torch.testing.assert_close(output, expected_output)
    elif isinstance(inputs[0], np.ndarray):
        np.testing.assert_allclose(output, expected_output)
    else:
        assert output == expected_output


def test_merge_numerics_but_not_lists():
    assert merge_numerics([(1, 2), (1, 2)], merge_lists=False) == [(1, 2), (1, 2)]
    assert merge_numerics([(1, 2), (1, 2)], merge_lists=True) == [1, 2, 1, 2]


@pytest.mark.parametrize(
    "inputs,expected_output",
    [
        (
            [
                tio.ScalarImage(tensor=torch.ones(1, 1, 1, 1), affine=np.eye(4, 4) * 2),
                tio.ScalarImage(
                    tensor=torch.ones(1, 1, 1, 1) * 2, affine=np.eye(4, 4) * 2
                ),
            ],
            tio.ScalarImage(
                tensor=torch.tensor([[[[1.0]]], [[[2.0]]]]), affine=np.eye(4, 4) * 2
            ),
        ),
        (
            [
                tio.LabelMap(tensor=torch.ones(1, 1, 1, 1)),
                tio.LabelMap(tensor=torch.ones(1, 1, 1, 1) * 2),
            ],
            tio.LabelMap(tensor=torch.tensor([[[[1.0]]], [[[2.0]]]])),
        ),
        (
            [
                tio.ScalarImage(tensor=torch.ones(1, 1, 1, 1)),
                tio.LabelMap(tensor=torch.ones(1, 1, 1, 1) * 2),
            ],
            tio.ScalarImage(tensor=torch.tensor([[[[1.0]]], [[[2.0]]]])),
        ),
        (
            [
                tio.ScalarImage(tensor=torch.ones(1, 1, 1, 1), affine=np.eye(4, 4) * 2),
                tio.ScalarImage(tensor=torch.ones(1, 1, 1, 1), affine=np.eye(4, 4)),
            ],
            "Trying to concatenate images with different voxel spacings!",
        ),
        (
            [
                tio.ScalarImage(tensor=torch.ones(1, 1, 1, 1)),
                tio.ScalarImage(tensor=torch.ones(1, 1, 1, 2)),
            ],
            "Trying to concatenate images with different spatial shapes!",
        ),
    ],
)
def test_concat_images(inputs, expected_output):
    if isinstance(expected_output, str):
        with pytest.raises(RuntimeError, match=expected_output):
            concat_images(inputs)
    else:
        output = concat_images(inputs)
        assert isinstance(output, type(expected_output))
        torch.testing.assert_close(output.tensor, expected_output.tensor)
        np.testing.assert_equal(output.affine, expected_output.affine)
