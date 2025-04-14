from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd
from monai.metrics.metric import Metric as MonaiMetric

from clinicadl.losses.utils import Loss
from clinicadl.metrics import ImplementedMetric
from clinicadl.metrics.config import MetricConfig
from clinicadl.metrics.config.base import LossMetricConfig
from clinicadl.metrics.factory import get_metric_config, get_metric_from_config
from clinicadl.tsvtools.utils import df_to_tsv, remove_non_empty_dir, tsv_to_df
from clinicadl.utils.json import read_json, write_json

MetricsTypes = Union[MonaiMetric, MetricConfig, ImplementedMetric, str]

LOSS = "Loss"


class GroupMetrics:
    def __init__(
        self,
        metrics: Union[MetricsTypes, list[MetricsTypes]],
        selection_metrics: Union[MetricsTypes, list[MetricsTypes]] = LOSS,
    ):
        self.metrics = self.check_metrics(metrics)

        if ImplementedMetric.LOSS not in self.metrics:
            self.metrics.append(ImplementedMetric.LOSS)

        self.selection_metrics = self.check_metrics(selection_metrics)

        if not set(self.selection_metrics).issubset(set(self.metrics)):
            raise ValueError(
                f"Selection metrics ({self.selection_metrics}) must be one of the provided metrics ({self.metrics})."
            )

        self.df = self._init_df()

    def get_loss(self, epoch: Optional[int] = None):
        if epoch:
            return self.df.at[epoch, LOSS]
        else:
            return self.df[LOSS].iloc[-1]

    def aggregate(self, epoch: int = 0):
        for metric, callable_metric in self._callable_metrics.items():
            value = callable_metric.aggregate()
            self.df.at[epoch, metric] = value.item()

    def reset(self):
        self._callable_metrics = {}
        for metric in self.metrics:
            if metric.value == LOSS:
                self._callable_metrics[metric.value] = self._callable_loss
            else:
                callable_metric, _ = get_metric_from_config(get_metric_config(metric))
                self._callable_metrics[metric.value] = callable_metric

    def _init_df(self):
        df = pd.DataFrame(
            columns=["epoch", "time"] + [metric.value for metric in self.metrics]
        )
        df.set_index(["epoch"], inplace=True)
        df.at[0, "time"] = 0.0
        df.at[0, LOSS] = 1

        return df

    def get_value(self, epoch: int, metric: str):
        try:
            return self.df.at[epoch, metric]
        except KeyError:
            raise KeyError(f"Metric '{metric}' not found in the provided metrics.")

    @staticmethod
    def check_metrics(
        metrics: Union[MetricsTypes, list[MetricsTypes]],
    ) -> list[ImplementedMetric]:
        """Check that all metrics are of the correct type and have the required attributes."""
        if isinstance(metrics, MetricsTypes):
            metrics = [metrics]

        if len(metrics) == 0:
            raise ValueError("At least one metric must be provided.")

        metrics_list = []

        # Check that all MetricConfig instances have the required attributes
        for metric in metrics:
            if isinstance(metric, MonaiMetric):
                ImplementedMetric._missing_(
                    metric.__class__.__name__
                )  # raise an error if missing
                metric = ImplementedMetric(metric.__class__.__name__)

            elif isinstance(metric, str):
                metric = ImplementedMetric(metric)

            elif isinstance(metric, MetricConfig):
                metric = metric.name

            elif not isinstance(metric, ImplementedMetric):
                raise ValueError(f"Metric '{metric}' is not an implemented metric.")

            metrics_list.append(metric)

        if not all(isinstance(metric, ImplementedMetric) for metric in metrics_list):
            raise ValueError(
                "All metrics must be implemented in ClinicaDL in order to be used."
            )

        return metrics_list

    def set_loss(self, loss: Loss):
        self._callable_loss, _ = get_metric_from_config(LossMetricConfig(loss_fn=loss))

    def on_train_end(self):
        pass

    def model_dump(self):
        dict_ = {}
        dict_["metrics"] = {
            "metrics": self.metrics,
            "selection_metrics": self.selection_metrics,
        }
        return dict_


class Metrics:
    def __init__(
        self,
        metrics: Union[MetricsTypes, list[MetricsTypes]],
        selection_metrics: Union[MetricsTypes, list[MetricsTypes]] = LOSS,
        compute_train_metrics: bool = True,
    ):
        self.train = GroupMetrics(metrics=metrics, selection_metrics=selection_metrics)
        self.val = GroupMetrics(metrics=metrics, selection_metrics=selection_metrics)

        self.training_loss = self._init_training_df()

        self.compute_train_metrics = compute_train_metrics

    @classmethod
    def from_json(cls, json_path: Path) -> Metrics:
        """
        Reads the JSON file and returns a Metrics object.
        """
        dict_ = read_json(json_path)
        return cls.from_dict(dict_)

    @classmethod
    def from_dict(cls, dict_: dict):
        metrics_config = dict_["metrics"]
        metrics = metrics_config["metrics"]
        selection_metrics = metrics_config["selection_metrics"]

        return cls(metrics=metrics, selection_metrics=selection_metrics)

    def write_training_loss(self, epoch: int, batch: int, loss: float):
        self.training_loss.at[(epoch, batch), LOSS] = loss

    def _init_training_df(self):
        df = pd.DataFrame(columns=["epoch", "batch", "time"])
        df.set_index(["epoch", "batch"], inplace=True)
        df.at[(0, 0), "time"] = 0.0
        df.at[(0, 0), LOSS] = 1

        return df

    def set_loss(self, loss: Loss):
        self.train.set_loss(loss)
        self.val.set_loss(loss)

    def model_dump(self):
        return self.val.model_dump()

    def write_json(self, json_path: Path, overwrite: bool = False) -> None:
        """
        Writes the serialized config class to a JSON file.
        """
        write_json(json_path=json_path, data=self.model_dump(), overwrite=overwrite)


# class RetainBest:
#     """
#     A class to retain the best and overfitting values for a set of wanted metrics.
#     """

#     def __init__(self, selection_metrics: List[str], n_classes: int = 0):
#         self.selection_metrics = selection_metrics

#         if LOSS in selection_metrics:
#             selection_metrics.remove(LOSS)
#             metric_module = MetricModule(selection_metrics)
#             selection_metrics.append(LOSS)
#         else:
#             metric_module = MetricModule(selection_metrics)

#         implemented_metrics = set(metric_optimum.keys())
#         if not set(self.selection_metrics).issubset(implemented_metrics):
#             raise NotImplementedError(
#                 f"The selection metrics {self.selection_metrics} are not all implemented. "
#                 f"Available metrics are {implemented_metrics}."
#             )
#         self.best_metrics = dict()
#         for selection in self.selection_metrics:
#             if n_classes > 2:
#                 metric_fn = metric_module.metrics[selection]
#                 metric_args = list(metric_fn.__code__.co_varnames)
#                 if "class_number" in metric_args:
#                     for class_number in range(n_classes):
#                         self.set_optimum(f"{selection}-{class_number}")
#                 else:
#                     self.set_optimum(selection)
#             else:
#                 self.set_optimum(selection)

#     def set_optimum(self, selection: str):
#         if metric_optimum[selection] == "min":
#             self.best_metrics[selection] = np.inf
#         elif metric_optimum[selection] == "max":
#             self.best_metrics[selection] = -np.inf
#         else:
#             raise ValueError(
#                 f"Objective {metric_optimum[selection]} unknown for metric {selection}."
#                 f"Please choose between 'min' and 'max'."
#             )

#     def step(self, metrics_valid: Dict[str, float]) -> Dict[str, bool]:
#         """
#         Computes for each metric if this is the best value ever seen.

#         Args:
#             metrics_valid: metrics computed on the validation set
#         Returns:
#             metric is associated to True if it is the best value ever seen.
#         """

#         metrics_dict = dict()
#         for selection in self.selection_metrics:
#             if metric_optimum[selection] == "min":
#                 metrics_dict[selection] = (
#                     metrics_valid[selection] < self.best_metrics[selection]
#                 )
#                 self.best_metrics[selection] = min(
#                     metrics_valid[selection], self.best_metrics[selection]
#                 )

#             else:
#                 metrics_dict[selection] = (
#                     metrics_valid[selection] > self.best_metrics[selection]
#                 )
#                 self.best_metrics[selection] = max(
#                     metrics_valid[selection], self.best_metrics[selection]
#                 )

#         return metrics_dict
