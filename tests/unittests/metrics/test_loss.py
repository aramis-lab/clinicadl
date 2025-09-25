import re

import pytest
import torch
import torchio as tio
from monai.metrics import LossMetric
from torch.nn import MSELoss

from clinicadl.data.dataloader import Batch
from clinicadl.data.structures import DataPoint
from clinicadl.metrics.config import LossMetricConfig
from clinicadl.metrics.monai_wrapper import MonaiMetricWrapper
from clinicadl.utils.exceptions import ClinicaDLArgumentError
from tests.unittests.resources.objects import MODEL


def test_loss_metric():
    config = LossMetricConfig(loss_key="loss_", reduction="mean")
    with pytest.raises(
        ClinicaDLArgumentError,
        match=(
            re.escape(
                "In LossMetricConfig, loss_key='loss_' but there is no such loss (returned by the 'get_loss_functions' method of you ClinicaDLModel). "
                "Losses are: ['loss']"
            )
        ),
    ):
        config.get_object(MODEL)

    MODEL.loss = MSELoss(reduction="sum")
    config = LossMetricConfig(loss_key="loss")
    assert isinstance(config.get_object(MODEL), MonaiMetricWrapper)
    assert isinstance(config.get_object(MODEL).metric, LossMetric)
    assert config.optimum() == "min"
    batch = Batch(
        [
            DataPoint(
                image=tio.ScalarImage(tensor=torch.ones(1, 1, 1, 1)),
                label=float(i),
                output=float(i + 1),
                participant=i,
                session=i,
            )
            for i in range(3)
        ]
    )
    metric: LossMetric = config.get_object(MODEL)
    monai_metric = metric.metric
    assert monai_metric.reduction == "sum"
    assert monai_metric.loss_fn.reduction == "none"
    out = metric(batch)
    assert out.shape == (3,)

    MODEL.loss = lambda x: x
    config = LossMetricConfig(loss_key="loss", reduction="mean", label_key=None)
    with pytest.raises(
        ClinicaDLArgumentError,
        match=re.escape(
            "The loss 'loss' (returned by the 'get_loss_functions' method of you ClinicaDLModel) "
            "doesn't have a 'reduction' attribute, so ClinicaDL can't compute the validation loss at the image level.",
        ),
    ):
        config.get_object(MODEL)
