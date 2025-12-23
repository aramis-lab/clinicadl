import shutil
from typing import TYPE_CHECKING, Any, Union

import pandas as pd
from pydantic import PositiveInt

from clinicadl.metrics.enum import Optimum
from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.dictionary.words import EPOCH
from clinicadl.utils.objects import HasConfig

from ..base import Callback

if TYPE_CHECKING:
    from clinicadl.io import Maps
    from clinicadl.train import TrainerState


class ModelCheckpointCallbackConfig(ObjectConfig["ModelCheckpointCallback"]):
    """Config class for ``ModelCheckpointCallback``."""

    metric: str
    epochs: list[PositiveInt]


class ModelCheckpointCallback(Callback):
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

        Parameters
        ----------
        metrics : str or list of str
            Name(s) of the metric(s) to monitor for model selection. These should match
            keys present in the `MetricsHandler` dictionary. If a single string is provided,
            it is converted to a list internally.
    """

    def __init__(
        self,
        metric: str,
        epochs: Sequence[int],
        mode: Union[Mode, Sequence[Mode]] = Mode.MIN,
    ):
        self._activated = False  # to prevent from calling in validation only

    # pylint: disable=arguments-differ, unused-argument
    def on_train_begin(self, maps: Maps, state: TrainerState, **kwargs) -> None:
        for metric in self.metric:
            maps.training.splits[state.split_idx].models.best_models(metric=metric)
        self._activated = True

    def on_validation_end(
        self, *, state: TrainerState, metrics: pd.DataFrame, **kwargs
    ) -> None:
        if not self._activated:
            return

    def on_epoch_end(self, maps: Maps, state: TrainerState, **kwargs) -> None:
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

    def _get_value(self, metrics: pd.DataFrame, state: TrainerState) -> float:
        """Gets the metric value."""
        assert (
            self.config.metric in metrics
        ), f"'{self.config.metric}' not found in the validation metrics!"

        value = metrics.set_index(EPOCH).loc[state.current_epoch, self.config.metric]

        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Value for metric '{self.config.metric}' at epoch {state.current_epoch} is not numeric."
            ) from exc

        return value
