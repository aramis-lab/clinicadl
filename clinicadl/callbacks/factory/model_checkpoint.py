import shutil
from typing import Union

import torch
from monai.metrics.metric import CumulativeIterationMetric as MonaiMetric

from clinicadl.dictionary.words import BATCH, EPOCH, LOSS, TIME
from clinicadl.losses.config import LossConfig
from clinicadl.losses.types import Loss
from clinicadl.maps.maps import Maps
from clinicadl.metrics.config.enum import Optimum
from clinicadl.metrics.metrics import (
    ClinicaDLMetrics,
    CustomMetric,
    LossMetricConfig,
    MetricConfig,
    Metrics,
)
from clinicadl.model import ClinicaDLModel
from clinicadl.trainer.config import _TrainingConfig
from clinicadl.utils.config import ClinicaDLConfig

from .base import Callback


class ModelCheckpoint(Callback, Metrics):
    def __init__(
        self,
        metrics: list[Union[MetricConfig, CustomMetric, MonaiMetric, Loss, LossConfig]],
    ):
        self.metrics = self.check_metrics(metrics)

    def on_epoch_end(self, config: _TrainingConfig, **kwargs):
        model_weights = {
            "model": config.model.network.state_dict(),
            EPOCH: config.epoch,
        }
        checkpoint_path = (
            config.maps.splits[config.split].tmp.path / "checkpoint.pth.tar"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_weights, checkpoint_path)

        for metric in self.metrics:
            metric_path = (
                config.maps.splits[config.split].best_metrics[metric.name].path
            )
            metric_path.mkdir(parents=True, exist_ok=True)

            optimum = metric.optimum()

            if (
                config.epoch == 0
                or (
                    optimum == Optimum.MAX
                    and (
                        config.metrics.df.at(config.epoch, metric.name)
                        > config.metrics.df.at(config.epoch - 1, metric.name)
                    )
                )
                or (
                    optimum == Optimum.MIN
                    and (
                        config.metrics.df.at(config.epoch, metric.name)
                        < config.metrics.df.at(config.epoch - 1, metric.name)
                    )
                )
            ):
                shutil.copyfile(checkpoint_path, metric_path / "model.pth.tar")
