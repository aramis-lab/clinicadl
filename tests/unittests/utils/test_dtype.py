import numpy as np
import pytest
import torch

from clinicadl.utils.dtype import read_dtype


@pytest.mark.parametrize(
    "input_,output",
    [
        ("torch.int32", torch.int32),
        ("np.float64", np.float64),
        ("numpy.int16", np.int16),
        ("float", "float"),
    ],
)
def test_read_dtype(input_, output):
    assert read_dtype(input_) == output
