from copy import deepcopy

import torch
import torch.nn as nn
import torchio as tio

from clinicadl.data.dataloader import Batch
from clinicadl.data.structures import DataPoint
from clinicadl.infer import Inferer

from .utils import NnWrapper


class MyInferer(Inferer):
    def __call__(
        self,
        x,
        network,
        input_dtype=None,
        **kwargs,
    ):
        image = self._get_input_tensor(x, input_dtype=torch.half)
        output = network(image, **kwargs)
        if isinstance(x, list):
            for x_, out_ in zip(x, output):
                x_["output"] = out_
        else:
            x["output"] = output

        return x


def test_inferer():
    inferer = MyInferer()
    sample = DataPoint(
        image=tio.ScalarImage(tensor=torch.randn(1, 3, 3, 3)),
        participant="abc",
        session="abc",
    )
    network = NnWrapper(nn.Sequential(nn.Flatten(), nn.Linear(3**3, 1)))
    network.to(dtype=torch.half)

    with torch.no_grad():
        out = inferer(
            sample,
            network,
        )
    assert out.participant == "abc"
    assert out["output"].shape == (1, 1)
    assert out is sample

    # kwargs
    with torch.no_grad():
        out_ = inferer(
            deepcopy(sample),
            network,
            offset=1,
        )
    torch.testing.assert_close(out["output"] + 1, out_["output"])

    # batch
    batch = Batch([sample, deepcopy(sample)])
    with torch.no_grad():
        out = inferer(
            batch,
            network,
        )
    assert out[0].participant == "abc"
    assert out[0]["output"].shape == (1,)
    assert out[0] is sample
