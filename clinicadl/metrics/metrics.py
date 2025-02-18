from typing import Optional, Union

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
    def __init__(self, metrics: list[Union[Metric, Loss]]):
        self.metrics = metrics
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.reset()

    def compute(
        self,
        batch: int,
        epoch: int,
        data: tuple[torch.Tensor, torch.Tensor],
        amp: bool = True,
    ):
        outputs, labels = data

        for metric in self.metrics:
            with autocast(
                device_type=self.device.type, enabled=amp
            ):  # TODO: check autocast for all metrics
                self.df[(batch, epoch), metric.__str__()] = metric(outputs, labels)

    def reset(self):
        self.df = pd.DataFrame(
            columns=["batch", "epoch"] + [metric.__str__() for metric in self.metrics]
        )
        self.df.set_index(["batch", "epoch"], inplace=True)

    def get_value(
        self, metric: Optional[Metric] = None, batch: int = 0, epoch: int = 0
    ):
        if len(self.metrics) == 1:
            return self.df.at[(batch, epoch), self.metrics[0].__str__()]
        elif len(self.metrics) > 1:
            if metric is None:
                raise ValueError(
                    "Please specify the metric you want to get the value from"
                )
            else:
                return self.df.at[(batch, epoch), metric.__str__()]
        else:
            raise ValueError("No metrics have been computed yet")


class TrainingMetrics:
    def __init__(self, metrics: list[Metric], loss: Loss):
        self.train_metrics = Metrics(metrics)
        self.val_metrics = Metrics(metrics)
        self.train_loss = Metrics([loss])
        self.val_loss = Metrics([loss])

    def compute(
        self,
        batch: int,
        epoch: int,
        data: tuple[torch.Tensor, torch.Tensor],
        train: bool = True,
        val: bool = True,
        loss: bool = True,
    ):
        if train:
            self.train_metrics.compute(batch, epoch, data)
            if loss:
                self.train_loss.compute(batch, epoch, data)
        if val:
            self.val_metrics.compute(batch, epoch, data)
            if loss:
                self.val_loss.compute(batch, epoch, data)
