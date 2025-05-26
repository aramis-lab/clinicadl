from typing import Union

from monai.metrics.metric import CumulativeIterationMetric as MonaiMetric

from clinicadl.losses.config import LossConfig
from clinicadl.losses.types import Loss
from clinicadl.metrics.metrics import CustomMetric, LossMetricConfig, MetricConfig
from clinicadl.utils.config import ClinicaDLConfig

from .base import Callback


class ModelCheckpoint(Callback):
    def __init__(
        self,
        metrics: list[Union[MetricConfig, CustomMetric, MonaiMetric, Loss, LossConfig]],
    ):
        self.metrics = self.check_metrics(metrics)

    def check_metrics(
        self, metrics: list[Union[MetricConfig, MonaiMetric, LossConfig, Loss]]
    ):
        if not isinstance(metrics, list):
            metrics = [metrics]
        for i, metric in enumerate(metrics):
            if isinstance(metric, LossConfig):
                metrics[i] = LossMetricConfig(loss_fn=metric.get_object())
            elif isinstance(metric, Loss):
                metrics[i] = LossMetricConfig(loss_fn=metric)
        return metrics

    def _save_tmp_weights(self, split: int):
        model_weights = {
            "model": self.model.network.state_dict(),
            EPOCH: self.epoch,
        }
        checkpoint_path = self.maps.splits[split].tmp.path / "checkpoint.pth.tar"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_weights, checkpoint_path)

        for metric in self.metrics:
            metric_path = maps.splits[split].best_metrics[name].path
            metric_path.mkdir(parents=True, exist_ok=True)

            optimum = metric_config.optimum()

            if (
                self.epoch == 0
                or (
                    optimum == Optimum.MAX
                    and (
                        self.metrics.get_value(self.epoch, name)
                        > self.metrics.get_value(self.epoch - 1, name)
                    )
                )
                or (
                    optimum == Optimum.MIN
                    and (
                        self.metrics.get_value(self.epoch, name)
                        < self.metrics.get_value(self.epoch - 1, name)
                    )
                )
            ):
                shutil.copyfile(checkpoint_path, metric_path / "model.pth.tar")

    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_epoch_begin(self, epoch: int, **kwargs):
        pass

    def on_epoch_end(self, epoch: int, **kwargs):
        pass

    def on_batch_begin(self, batch: int, **kwargs):
        pass

    def on_batch_end(self, batch: int, **kwargs):
        pass

    def on_backward_begin(self, **kwargs):
        pass

    def on_validation_begin(self, **kwargs):
        pass

    def on_validation_end(self, **kwargs):
        pass
