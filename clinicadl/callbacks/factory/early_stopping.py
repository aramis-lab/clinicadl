import math
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch

from clinicadl.callbacks.training_state import _TrainingState
from clinicadl.dictionary.suffixes import JSON
from clinicadl.utils.json import read_json, write_json

from .base import Callback

logger = getLogger("clinicadl.early_stopping")


class Mode(str, Enum):
    """Supported mode for Early Stopping."""

    MIN = "min"
    MAX = "max"


class OneMetricEarlyStopping(Callback):
    """
    Early stopping for a single metric.

    Parameters
    ----------
    metric : str
        Name of the metric to monitor.
    patience : int
        Number of epochs to wait for improvement before stopping.
    min_delta : float, optional (default=0.0)
        Minimum change in the monitored metric to qualify as an improvement.
    mode : Mode, optional (default=Mode.MIN)
        Whether the metric should be minimized ('min') or maximized ('max').
    check_finite : bool, optional (default=True)
        If True, stop training if metric value is NaN or infinite.
    upper_bound : float, optional
        If metric goes above this value, training stops.
    lower_bound : float, optional
        If metric goes below this value, training stops.

    """

    def __init__(
        self,
        metric: str,
        patience: int,
        min_delta: Optional[float] = 0.0,
        mode: Mode = Mode.MIN,
        check_finite: bool = True,
        upper_bound: Optional[float] = None,
        lower_bound: Optional[float] = None,
    ) -> None:
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.check_finite = check_finite
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self._check_bounds()
        self.is_better = self._get_comparison_function()
        self.reset()

    def _get_comparison_function(self):
        """Return the function to compare current and best metric values."""
        if self.mode == Mode.MIN:
            return lambda value, best: value < best - self.min_delta
        if self.mode == Mode.MAX:
            return lambda value, best: value > best + self.min_delta
        raise ValueError(f"Unknown mode: {self.mode}")

    def _check_bounds(self):
        """Validate that upper_bound is greater than lower_bound."""
        if self.upper_bound is not None and self.lower_bound is not None:
            if self.lower_bound > self.upper_bound:
                raise ValueError("Upper bound should be greater than lower bound.")

    def reset(self) -> None:
        """Reset the best metric and bad epoch counter."""
        if self.mode == Mode.MIN:
            self.best = np.inf
        elif self.mode == Mode.MAX:
            self.best = -np.inf
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.num_bad_epochs = 0

    def should_stop_training(self, config: _TrainingState) -> bool:
        """
        Check if training should stop at the end of an epoch.

        Parameters
        ----------
        config : _TrainingState
            Current training state, must contain metrics DataFrame.

        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """
        df = config.metrics.df

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Metrics DataFrame is missing or invalid.")

        if self.metric not in df.columns:
            raise ValueError(
                f"Metric '{self.metric}' not found in metrics DataFrame columns."
            )
        if config.epoch not in df.index:
            raise ValueError(
                f"Epoch {config.epoch} not found in metrics DataFrame index."
            )

        value = df.at[config.epoch, self.metric]
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Value for metric '{self.metric}' at epoch {config.epoch} is not numeric."
            ) from exc

        if pd.isna(value):
            raise ValueError(
                f"Metric '{self.metric}' value at epoch {config.epoch} is NaN."
            )

        if self.check_finite and (math.isinf(value) or math.isnan(value)):
            logger.warning(
                "Metric '%s' value at epoch %s is not finite. Stopping training.",
                self.metric,
                config.epoch,
            )
            return True

        if self.upper_bound is not None and (value > self.upper_bound):
            logger.warning(
                "Metric '%s' value %s  exceeded upper bound %s. Stopping training.",
                self.metric,
                value,
                self.upper_bound,
            )
            return True

        if self.lower_bound is not None and value < self.lower_bound:
            logger.warning(
                "Metric '%s' value %s fell below lower bound %s. Stopping training.",
                self.metric,
                value,
                self.lower_bound,
            )
            return True

        if self.is_better(value, self.best):
            self.num_bad_epochs = 0
            self.best = value
        else:
            self.num_bad_epochs += 1
            logger.debug(
                "No improvement in '%s' for %s epochs.",
                self.metric,
                self.num_bad_epochs,
            )

        if self.patience is not None and self.num_bad_epochs >= self.patience:
            logger.info(
                "Early stopping triggered on metric '%s' after %s epochs without improvement.",
                self.metric,
                self.num_bad_epochs,
            )
            return True

        return False

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the callback to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the callback.
        """
        json_dict = super().to_dict()
        json_dict.update(
            {
                "metrics": self.metric,
                "patience": self.patience,
                "min_delta": self.min_delta,
                "mode": self.mode,
                "check_finite": self.check_finite,
                "upper_bound": self.upper_bound,
                "lower_bound": self.lower_bound,
            }
        )
        return json_dict


class EarlyStopping(Callback):
    """
    Early stopping callback monitoring one or multiple metrics.

    This callback stops training early if monitored metric(s) do not improve for a
    specified number of epochs. It can monitor multiple metrics simultaneously and
    supports individual configurations per metric.

    Parameters
    ----------
        metrics : str or list of str
            Metric(s) to monitor.
        patience : int or list of int, optional
            Number of epochs with no improvement after which training will be stopped.
            If a single int is provided, it is applied to all metrics.
        min_delta : float or list of float, optional (default=0.0)
            Minimum change in monitored metric to qualify as an improvement.
        mode : Mode or list of Mode, optional (default=Mode.MIN)
            Whether to minimize or maximize the metric (e.g., loss vs. accuracy).
        check_finite : bool or list of bool, optional (default=True)
            Whether to stop if the metric becomes NaN or infinite.
        upper_bound : float or list of float, optional
            Optional upper threshold that will trigger stopping if exceeded.
        lower_bound : float or list of float, optional
            Optional lower threshold that will trigger stopping if dropped below.


    .. note::

        Behavior regarding interaction with ``ModelSelection``:

        - If neither ``EarlyStopping`` nor ``ModelSelection`` are used:
          the final model and the best-loss model are saved, but no early stopping is applied.
        - If ``EarlyStopping`` is used without ``ModelSelection``:
          training stops when all monitored metrics stop improving.
          For each metric, a ``ModelSelection`` object is automatically created.
        - If ``ModelSelection`` is used without ``EarlyStopping``:
          best models are saved based on monitored metrics, but training completes all epochs.
        - If both are used:
          ``EarlyStopping`` metrics are automatically tracked by ``ModelSelection``,
          ensuring best-performing models are saved.

    .. warning::

        Multiple ``EarlyStopping`` callbacks can be registered simultaneously.
        In such cases, training stops as soon as *any* of them triggers its stopping criterion.

    Examples
    --------
    .. code-block:: python

        from clinicadl.callbacks import EarlyStopping

        early_stopping = EarlyStopping(metrics="mae", patience=5)

        trainer = Trainer(
            maps_path="maps",
            callbacks=[early_stopping]
        )
    """

    def __init__(
        self,
        metrics: Union[str, list[str]],
        patience: Optional[Union[int, list[int]]] = None,
        min_delta: Optional[Union[float, list[float]]] = 0.0,
        mode: Union[Mode, list[Mode]] = Mode.MIN,
        check_finite: Union[bool, list[bool]] = True,
        upper_bound: Optional[Union[float, list[float]]] = None,
        lower_bound: Optional[Union[float, list[float]]] = None,
    ) -> None:
        self.metrics = metrics if isinstance(metrics, list) else [metrics]
        len_metrics = len(metrics if isinstance(metrics, list) else [metrics])

        def check_list(value) -> list:
            if not isinstance(value, list):
                list_ = [value]
            else:
                list_ = value

            if len(list_) != 1 and len(list_) != len_metrics:
                raise ValueError(
                    f"List {list_} must have the same length as metrics: {len_metrics}"
                )
            elif len(list_) == 1:
                list_ = list_ * len_metrics
            return list_

        self.patience = check_list(patience)
        self.min_delta = check_list(min_delta)
        self.mode = check_list(mode)
        self.check_finite = check_list(check_finite)
        self.upper_bound = check_list(upper_bound)
        self.lower_bound = check_list(lower_bound)

        self.early_stoppers: list[OneMetricEarlyStopping] = []

        for i, metric in enumerate(self.metrics):
            self.early_stoppers.append(
                OneMetricEarlyStopping(
                    metric=metric,
                    patience=self.patience[i],
                    min_delta=self.min_delta[i],
                    mode=self.mode[i],
                    check_finite=self.check_finite[i],
                    upper_bound=self.upper_bound[i],
                    lower_bound=self.lower_bound[i],
                )
            )

    def on_epoch_end(self, config: _TrainingState, **kwargs) -> None:
        """
        Called at the end of each epoch.

        Updates `config.stop` to True if all monitored metrics meet early stopping criteria.
        """
        should_stops = [
            metric.should_stop_training(config) for metric in self.early_stoppers
        ]
        should_stop = all(should_stops)
        if should_stop:
            logger.info(
                "Early stopping criteria met for all monitored metrics. Stopping training."
            )
        config.stop = should_stop

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the callback to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the callback.
        """
        json_dict = super().to_dict()
        json_dict.update(
            {
                "metrics": self.metrics,
                "patience": self.patience,
                "min_delta": self.min_delta,
                "mode": self.mode,
                "check_finite": self.check_finite,
                "upper_bound": self.upper_bound,
                "lower_bound": self.lower_bound,
            }
        )
        return json_dict

    def save_checkpoint(
        self,
        checkpoint_path: Path,
        **kwargs,
    ) -> None:
        """To save the state of the early stoppers."""
        checkpoints = {}
        for stopper in self.early_stoppers:
            checkpoints[stopper.metric] = {
                "best": stopper.best,
                "num_bad_epochs": stopper.num_bad_epochs,
            }
        filename = checkpoint_path.with_suffix(JSON)
        write_json(filename, checkpoints)

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        **kwargs,
    ) -> None:
        """To load a checkpoint saved with 'save_checkpoint'."""
        filename = checkpoint_path.with_suffix(JSON)
        checkpoint: dict = read_json(filename)
        for metric, stopper in zip(self.metrics, self.early_stoppers):
            stopper.best = checkpoint[metric]["best"]
            stopper.num_bad_epochs = checkpoint[metric]["num_bad_epochs"]
