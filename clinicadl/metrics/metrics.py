from typing import Callable, Optional, Union

import pandas as pd
import torch
from monai.metrics.metric import Metric as MonaiMetric
from monai.metrics.regression import MAEMetric, RMSEMetric, SSIMMetric
from torch.amp.autocast_mode import autocast

from clinicadl.losses.utils import Loss
from clinicadl.metrics.config.classification import (
    ConfusionMatrixMetricConfig,
    ROCAUCMetricConfig,
)
from clinicadl.metrics.factory import get_metric_from_config
from clinicadl.networks.factory import ImplementedNetwork, get_network_config
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.exceptions import ClinicaDLMetricsError

ClassificationLoss = ["CrossEntropyLoss", "MultiMarginLoss"]
ClassificationMetrics = [
    "BA",
    "accuracy",
    "F1_score",
    "sensitivity",
    "specificity",
    "PPV",
    "NPV",
    "MCC",
    "MK",
    "LR_plus",
    "LR_minus",
]


ReconstructionMetrics = ["MAE", "RMSE", "PSNR", "SSIM"]
ReconstructionLosses = [
    "L1Loss",
    "MSELoss",
    "KLDivLoss",
    "BCEWithLogitsLoss",
    "HuberLoss",
    "SmoothL1Loss",
    "VAEGaussianLoss",
    "VAEBernoulliLoss",
    "VAEContinuousBernoulliLoss",
]

RegressionMetrics = [RMSEMetric(), MAEMetric()]
RegressionLosses = [
    "L1Loss",
    "MSELoss",
    "KLDivLoss",
    "BCEWithLogitsLoss",
    "HuberLoss",
    "SmoothL1Loss",
]


class BaseMetrics:
    def __init__(self, metrics: list[Callable]):
        self.metrics = metrics
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.amp: bool = True
        self.reset()

    # def set_loss(self, loss: Loss):
    #     self.loss = loss

    def compute(
        self,
        batch: int,
        epoch: int,
        data: tuple[torch.Tensor, torch.Tensor],
        loss: Optional[torch.Tensor] = None,
    ):
        outputs, labels = data

        for metric in self.metrics:
            with autocast(
                device_type=self.device.type, enabled=self.amp
            ):  # TODO: check autocast for all metrics
                self.df.at[(epoch, batch), metric.__str__()] = (
                    metric(outputs, labels).mean().item()
                )
        if loss:
            self.df.at[(epoch, batch), "loss"] = loss.item()

    @property
    def loss(self):
        return self.get_loss()

    def get_loss(self, epoch: Optional[int] = None, batch: Optional[int] = None):
        return self.get_value("loss", epoch, batch)

    def get_value(
        self, metric: str, epoch: Optional[int] = None, batch: Optional[int] = None
    ):
        if epoch is not None and batch is not None:
            return self.df.at[(epoch, batch), metric]
        elif epoch is not None:
            return self.df.at[(epoch, "mean"), metric]
        else:
            return self.df.at[("mean", "mean"), metric]

    def reset(self):
        self.df = pd.DataFrame(
            columns=["epoch", "batch"]
            + [metric.__str__() for metric in (self.metrics + ["loss"])]
        )
        self.df.set_index(["epoch", "batch"], inplace=True)

        import numpy as np

        self.df.loc[("mean", "mean"), "loss"] = 10


class Metrics:
    def __init__(self, metrics: list[MonaiMetric], selection_metric: str = "loss"):
        self.metrics = metrics
        self.train = BaseMetrics(metrics)
        self.val = BaseMetrics(metrics)
        self.selection_metric = selection_metric

    def on_epoch_end(self, epoch: int):
        self.train.df.loc[(epoch, "mean"), :] = self.train.df.loc[epoch, :].mean()
        self.train.df.loc[("mean", "mean"), "loss"] = self.train.df.loc[
            (epoch, "mean"), "loss"
        ]

        self.val.df.loc[(epoch, "mean"), :] = self.val.df.loc[epoch, :].mean()
        self.val.df.loc[("mean", "mean"), "loss"] = self.val.df.loc[
            (epoch, "mean"), "loss"
        ]

        # # Calculer la moyenne pour l'epoch donné
        # mean_train_values = self.train.df.loc[epoch].mean()

        # # Ajouter la ligne "mean" pour cet epoch
        # self.train.df = pd.concat(
        #     [self.train.df, pd.DataFrame(mean_train_values).T.assign(epoch=epoch, batch="mean").set_index(["epoch", "batch"])]
        # )

        # # Ajouter la moyenne globale pour "loss"
        # self.train.df.loc[("mean", "mean"), "loss"] = self.train.df.xs("mean", level="batch")["loss"].mean()

        # # Calculer la moyenne pour l'epoch donné
        # mean_val_values = self.val.df.loc[epoch].mean()

        # # Ajouter la ligne "mean" pour cet epoch
        # self.val.df = pd.concat(
        #     [self.val.df, pd.DataFrame(mean_val_values).T.assign(epoch=epoch, batch="mean").set_index(["epoch", "batch"])]
        # )

        # # Ajouter la moyenne globale pour "loss"
        # self.val.df.loc[("mean", "mean"), "loss"] = self.val.df.xs("mean", level="batch")["loss"].mean()
