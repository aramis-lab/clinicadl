import pandas as pd
import torch
from monai.metrics.metric import Metric
from torch.amp.autocast_mode import autocast

from clinicadl.losses.utils import Loss
from clinicadl.metrics.config.classification import (
    ConfusionMatrixMetricConfig,
    ROCAUCMetricConfig,
)
from clinicadl.metrics.factory import get_metric_from_config
from clinicadl.networks.factory import ImplementedNetwork, get_network_config
from clinicadl.utils.config import ClinicaDLConfig


class Metrics:
    def __init__(self, metrics: list[Metric], loss: Loss):
        self.metrics = metrics
        self.loss = loss

        self.val_loss = 1
        self.train_loss = 1

        self.train_metrics = pd.DataFrame(
            columns=["batch", "epoch"]
            + [metric.__str__() for metric in self.metrics]
            + [self.loss.__str__()]
        )  # index = ["batch", "epoch"] ,columns = [metric.__str__() for metric in self.metrics] + [self.loss.__str__()]
        self.val_metrics = pd.DataFrame()  # index = ["batch", "epoch"] ,columns = [metric.__str__() for metric in self.metrics] + [self.loss.__str__()]  )

    def compute(
        self,
        batch: int,
        epoch: int,
        data: tuple[torch.Tensor, torch.Tensor],
        compute_loss: bool = True,
        val: bool = True,
        amp: bool = True,
    ):
        outputs, labels = data
        with autocast("cpu", enabled=amp):
            loss_value = self.loss(outputs, labels) if compute_loss else None

        for metric in self.metrics:
            if val:
                self.val_metrics.loc[(batch, epoch), metric.__str__()] = metric(
                    outputs, labels
                )
            else:
                self.train_metrics.loc[(batch, epoch), metric.__str__()] = metric(
                    outputs, labels
                )

        return loss_value, (self.val_metrics if val else self.train_metrics)
