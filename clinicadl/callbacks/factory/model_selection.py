import shutil
from typing import Any, Union

from clinicadl.callbacks.training_state import _TrainingState
from clinicadl.metrics.enum import Optimum

from .base import Callback


class ModelSelection(Callback):
    """
    Callback that manages model checkpoint selection based on specified metrics.

    At the end of each epoch, this callback evaluates the monitored metrics and saves
    the model checkpoint corresponding to the best score (either minimum or maximum,
    depending on the configured criterion).

    This ensures that the model associated with the best performance on each tracked
    metric is preserved and can be restored later.

    Attributes
    ----------
        metrics : list of str
            List of metric names used to determine whether a new best model should be saved.

    .. note::

        When both ``ModelSelection`` and ``EarlyStopping`` are used:

        - ``_CallbacksHandler`` ensures all metrics used by ``EarlyStopping`` are added
          to ``ModelSelection`` if not already present.
        - This guarantees that any model selected based on a stopping condition is also
          saved properly.

    .. note::

        - The logic for determining whether a metric has improved is based on whether it
          should be maximized or minimized (``Optimum.MAX`` or ``Optimum.MIN``).
        - Models are stored in separate folders per metric to avoid overwriting.

    Examples
    --------
    .. code-block:: python

        from clinicadl.callbacks import ModelSelection
        from clinicadl.metrics import MSEMetricConfig, MAEMetricConfig
        from clinicadl.trainer import Trainer

        metrics = {
            "mse_mean": MSEMetricConfig(reduction="mean"),
            "mse_sum": MSEMetricConfig(reduction="sum"),
            "mae": MAEMetricConfig()
        }

        selection = ModelSelection(metrics=["mse_mean", "mse_sum", "mae"])

        trainer = Trainer(
            maps_path="maps",
            metrics=metrics,
            callbacks=[selection]
        )
    """

    def __init__(
        self,
        metrics: Union[str, list[str]],
    ):
        """
        Parameters
        ----------
        metrics : str or list of str
            Name(s) of the metric(s) to monitor for model selection. These should match
            keys present in the `MetricsHandler` dictionary. If a single string is provided,
            it is converted to a list internally.
        """
        self.metrics = metrics if isinstance(metrics, list) else [metrics]

    def on_train_begin(self, config: _TrainingState, **kwargs) -> None:
        """
        Initialize storage structures for best metrics and create necessary folders.
        """

        # config.maps.training.create_split(config.split)
        for metric in self.metrics:
            config.maps.training.splits[config.split.index]._create_best_metrics(
                metric=metric
            )
        # config.split.write_json(config.maps.training.splits[config.split.index].caps_dataset_json)

    def on_epoch_end(self, config: _TrainingState, **kwargs) -> None:
        """
        At each epoch, check whether any metric has improved. If so, copy the current
        model and optimizer checkpoints into the best directory for that metric.
        """

        for metric in self.metrics:
            metric_dir = config.maps.training.splits[config.split.index].best_metrics[
                metric
            ]
            # metric_path.mkdir(parents=True, exist_ok=True)

            optimum = config.metrics.metrics[metric].optimum()

            if (
                config.epoch == 0
                or (
                    optimum == Optimum.MAX
                    and (
                        config.metrics.df.at[config.epoch, metric]
                        > config.metrics.df.at[config.epoch - 1, metric]
                    )
                )
                or (
                    optimum == Optimum.MIN
                    and (
                        config.metrics.df.at[config.epoch, metric]
                        < config.metrics.df.at[config.epoch - 1, metric]
                    )
                )
            ):
                tmp_dir = config.maps.training.splits[config.split.index].tmp.epochs[
                    config.epoch
                ]

                shutil.copyfile(tmp_dir.model, metric_dir.model)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the callback to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the callback.
        """
        json_dict = super().to_dict()
        json_dict.update({"metrics": self.metrics})
        return json_dict
