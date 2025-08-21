import numpy as np
import torch

from clinicadl.transforms.homemade import Format


def test_Format():
    transform = Format(dtype=torch.float64, squeeze=True)
    out = transform(np.array([[[1], [0]]]))
    assert out.shape == (2,)
    assert out.dtype == np.float64

    transform = Format(squeeze=2)
    out = transform(np.array([[[1], [0]]]))
    assert out.shape == (1, 2)

    transform = Format(squeeze=[0, 2])
    out = transform(np.array([[[1], [0]]]))
    assert out.shape == (2,)

    transform = Format(unsqueeze=1)
    out = transform(torch.tensor([0, 1], dtype=torch.int16))
    assert out.shape == (2, 1)
    assert out.dtype == torch.int16
