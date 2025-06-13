import shutil
from typing import Union

import torch
from monai.metrics.metric import CumulativeIterationMetric as MonaiMetric

from clinicadl.dictionary.suffixes import PTH, TAR
from clinicadl.dictionary.words import CHECKPOINT, EPOCH, MODEL
from clinicadl.losses.config import LossConfig
from clinicadl.losses.types import Loss
from clinicadl.metrics.config.enum import Optimum
from clinicadl.metrics.metrics import (
    CustomMetric,
    MetricConfig,
    Metrics,
)
from clinicadl.utils.config.training import _TrainingState

from .base import Callback


class ModelCheckpoint(Callback, Metrics):
    """TO COMPLETE"""

    def __init__(
        self,
        metrics: list[Union[MetricConfig, CustomMetric, MonaiMetric, Loss, LossConfig]],
    ):
        self.metrics = self.check_metrics(metrics)

    def on_train_begin(self, config: _TrainingState, **kwargs):
        """TO COMPLETE"""

        if config.split.train_loader is None:
            raise ValueError(
                "The split has no train_loader defined. Please run `get_dataloader()`"
            )
        if config.split.val_loader is None:
            raise ValueError(
                "The split has no val_loader defined. Please run `get_dataloader()`"
            )

        metrics_name = [
            metric.name
            for metric in self.metrics  # TODO: maybe we need to remove the metric of the name of the directory ?
        ]
        config.maps.create_split(config.split, metrics_name)
        config.split.write_json(config.maps.splits[config.split.index].split_json)

    def on_epoch_end(self, config: _TrainingState, **kwargs):
        """TO COMPLETE"""

        model_weights = {
            MODEL: config.model.network.state_dict(),
            EPOCH: config.epoch,
        }
        checkpoint_path = config.maps.splits[config.split.index].tmp.path / (
            CHECKPOINT + PTH + TAR
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_weights, checkpoint_path)

        for metric in self.metrics:
            metric_path = (
                config.maps.splits[config.split.index].best_metrics[metric.name].path
            )
            metric_path.mkdir(parents=True, exist_ok=True)

            optimum = metric.optimum()

            if (
                config.epoch == 0
                or (
                    optimum == Optimum.MAX
                    and (
                        config.metrics.df.at[config.epoch, metric.name]
                        > config.metrics.df.at[config.epoch - 1, metric.name]
                    )
                )
                or (
                    optimum == Optimum.MIN
                    and (
                        config.metrics.df.at[config.epoch, metric.name]
                        < config.metrics.df.at[config.epoch - 1, metric.name]
                    )
                )
            ):
                shutil.copyfile(checkpoint_path, metric_path / (MODEL + PTH + TAR))
