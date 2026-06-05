import re

import numpy as np
import pytest
import torch
import torchio as tio
from monai.metrics import LossMetric
from torch.nn import MSELoss

from clinicadl.data.structures import DataPoint
from clinicadl.metrics.config import LossMetricConfig
from clinicadl.metrics.monai_wrapper import MonaiMetricWrapper
from clinicadl.utils.exceptions import ClinicaDLArgumentError


class Batch(list):
    def get_field(self, key: str, **kwargs):
        if "image" in key:
            return torch.stack([data[key].tensor for data in self])
        return torch.Tensor([[data[key]] for data in self])


class Model:
    def get_loss_functions(self):
        return {"loss": MSELoss(reduction="sum")}


class ModelBis:
    def get_loss_functions(self):
        return {"loss": lambda x: x}


MODEL = Model()
MODEL_BIS = ModelBis()


def test_loss_metric():
    config = LossMetricConfig(loss_name="loss_", reduction="mean")
    with pytest.raises(
        ClinicaDLArgumentError,
        match=(
            re.escape(
                "In LossMetricConfig, loss_name='loss_' but there is no such loss (returned by the 'get_loss_functions' method of you Model). "
                "Losses are: ['loss']"
            )
        ),
    ):
        config.get_object(MODEL)

    config = LossMetricConfig(loss_name="loss")
    assert isinstance(config.get_object(MODEL), MonaiMetricWrapper)
    assert isinstance(config.get_object(MODEL).metric, LossMetric)
    assert config.optimum() == "min"
    batch = Batch(
        [
            DataPoint(
                image=tio.ScalarImage(tensor=torch.ones(1, 2, 2, 2)),
                label=float(i),
                output=float(i + 1),
                output_image=tio.ScalarImage(tensor=torch.ones(1, 2, 2, 2) + 3),
                participant_id=str(i),
                session_id=str(i),
            )
            for i in range(3)
        ]
    )
    metric: LossMetric = config.get_object(MODEL)
    monai_metric = metric.metric
    assert monai_metric.reduction == "sum"
    assert monai_metric.loss_fn.reduction == "none"
    out = metric(batch)
    torch.testing.assert_close(out, torch.tensor([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(metric.aggregate(), 3.0)

    config = LossMetricConfig(
        loss_name="loss", label_key="image", pred_key="output_image"
    )
    metric: LossMetric = config.get_object(MODEL)
    out = metric(batch)
    torch.testing.assert_close(out, torch.tensor([9 * 8] * 3, dtype=torch.float32))
    np.testing.assert_allclose(metric.aggregate(), 9 * 8 * 3)

    config = LossMetricConfig(loss_name="loss", reduction="mean", label_key=None)
    with pytest.raises(
        ClinicaDLArgumentError,
        match=re.escape(
            "The loss 'loss' (returned by the 'get_loss_functions' method of you Model) "
            "doesn't have a 'reduction' attribute, so ClinicaDL can't compute the validation loss at the image level.",
        ),
    ):
        config.get_object(MODEL_BIS)
