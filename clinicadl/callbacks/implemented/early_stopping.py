from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from enum import Enum
from logging import getLogger
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, TypeVar, Union

import numpy as np
import pandas as pd
from pydantic import Field, NonNegativeFloat, PositiveInt, model_validator
from typing_extensions import Self

from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.objects import HasConfig

from ..base import Callback
from .utils import get_metric_key_error, get_metric_value

if TYPE_CHECKING:
    from clinicadl.metrics import Metric
    from clinicadl.train import TrainerState


logger = getLogger("clinicadl.early_stopping")


class Mode(str, Enum):
    """Supported mode for Early Stopping."""

    MIN = "min"
    MAX = "max"


class _OneMetricEarlyStoppingConfig(ObjectConfig["_OneMetricEarlyStopping"]):
    """Config class for ``_OneMetricEarlyStopping``."""

    metric: str
    patience: PositiveInt
    min_delta: NonNegativeFloat
    mode: Optional[Mode]  # we may not know the mode at first
    check_finite: bool
    upper_bound: Optional[float]
    lower_bound: Optional[float]

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        """Validate that upper_bound is greater than lower_bound."""
        if self.upper_bound is not None and self.lower_bound is not None:
            if self.lower_bound > self.upper_bound:
                raise ValueError("Upper bound should be greater than lower bound.")

        return self

    @classmethod
    def _get_class(cls):
        return _OneMetricEarlyStopping


class _OneMetricEarlyStopping(HasConfig[_OneMetricEarlyStoppingConfig]):
    """
    Early stopping for a single metric.
    """

    _config_type = _OneMetricEarlyStoppingConfig

    def __init__(
        self,
        metric: str,
        patience: int,
        min_delta: float,
        mode: Mode,
        check_finite: bool,
        upper_bound: Optional[float],
        lower_bound: Optional[float],
    ) -> None:
        self.config = self._config_type(
            metric=metric,
            patience=patience,
            min_delta=min_delta,
            mode=mode,
            check_finite=check_finite,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
        )

        self.is_better = self._get_comparison_function()
        self.reset()

    def _get_comparison_function(self) -> Callable:
        """Return the function to compare current and best metric values."""
        if self.config.mode == Mode.MIN:
            return lambda value, best: value < best - self.config.min_delta
        elif self.config.mode == Mode.MAX:
            return lambda value, best: value > best + self.config.min_delta

    def reset(self) -> None:
        """Resets the best metric and counter."""
        if self.config.mode == Mode.MIN:
            self.best = np.inf
        elif self.config.mode == Mode.MAX:
            self.best = -np.inf

        self.num_bad_epochs = 0

    def step(self, metrics_df: pd.DataFrame, state: TrainerState) -> bool:
        """
        Check if training should stop at the end of an epoch.
        """
        value = get_metric_value(
            metrics_df, metric_name=self.config.metric, epoch=state.current_epoch
        )

        if self.config.check_finite and (math.isinf(value) or math.isnan(value)):
            logger.warning(
                "Metric '%s' value at epoch %s is not a finite float. Stopping training.",
                self.config.metric,
                state.current_epoch,
            )
            return True

        if self.config.upper_bound is not None and (value > self.config.upper_bound):
            logger.warning(
                "Metric '%s' value %s exceeds upper bound %s at epoch %s. Stopping training.",
                self.config.metric,
                value,
                self.config.upper_bound,
                state.current_epoch,
            )
            return True

        if self.config.lower_bound is not None and (value < self.config.lower_bound):
            logger.warning(
                "Metric '%s' value %s falls below lower bound %s at epoch %s. Stopping training.",
                self.config.metric,
                value,
                self.config.lower_bound,
                state.current_epoch,
            )
            return True

        if self.is_better(value, self.best):
            self.num_bad_epochs = 0
            self.best = value
        else:
            self.num_bad_epochs += 1
            logger.debug(
                "No improvement in '%s' for %s evaluation step(s).",
                self.config.metric,
                self.num_bad_epochs,
            )

        if self.num_bad_epochs >= self.config.patience:
            logger.info(
                "Early stopping triggered on metric '%s' after %s evaluation(s) without improvement.",
                self.config.metric,
                self.num_bad_epochs,
            )
            return True

        return False

    def state_dict(self) -> Mapping[str, Any]:
        return {"best": self.best, "num_bad_epochs": self.num_bad_epochs}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.best = state_dict["best"]
        self.num_bad_epochs = state_dict["num_bad_epochs"]


T = TypeVar("T")


class EarlyStoppingCallbackConfig(ObjectConfig["EarlyStoppingCallback"]):
    """Config class for ``EarlyStoppingCallback``."""

    stoppers: Sequence[_OneMetricEarlyStoppingConfig] = Field(
        reader=lambda stoppers: list(
            map(_OneMetricEarlyStoppingConfig.from_dict, stoppers)
        )
    )

    @classmethod
    def from_parameters(
        cls,
        metric: Union[str, Sequence[str]],
        patience: Union[int, Sequence[int]],
        min_delta: Union[float, Sequence[float]],
        check_finite: Union[bool, Sequence[bool]],
        upper_bound: Union[Optional[float], Sequence[Optional[float]]],
        lower_bound: Union[Optional[float], Sequence[Optional[float]]],
    ) -> Self:
        """
        Creates a sequence of Early Stoppers from sequences of parameters.
        """
        metric = (
            metric
            if isinstance(metric, Sequence) and not isinstance(metric, str)
            else [metric]
        )
        n = len(metric)
        configs = []
        for m, p, m_d, c_f, u_b, l_b in zip(
            metric,
            cls._ensure_sequence(patience, n, "patience"),
            cls._ensure_sequence(min_delta, n, "min_delta"),
            cls._ensure_sequence(check_finite, n, "check_finite"),
            cls._ensure_sequence(upper_bound, n, "upper_bound"),
            cls._ensure_sequence(lower_bound, n, "lower_bound"),
        ):
            configs.append(
                _OneMetricEarlyStoppingConfig(
                    metric=m,
                    patience=p,
                    min_delta=m_d,
                    check_finite=c_f,
                    upper_bound=u_b,
                    lower_bound=l_b,
                    mode=None,
                )
            )

        return cls(stoppers=configs)

    @classmethod
    def _ensure_sequence(
        cls, x: Union[T, Sequence[T]], len_: int, name: str
    ) -> Sequence[T]:
        """
        Ensure a sequence for any parameter.
        """
        if not isinstance(x, Sequence) or isinstance(x, str):
            return [x] * len_

        if len(x) == 1:
            return x * len_

        if len(x) != len_:
            raise ValueError(
                f"For {cls._get_name()}, there are {len_} metrics, but you passed {len(x)} '{name}': {x}"
            )

        return x

    @classmethod
    def _get_class(cls):
        return EarlyStoppingCallback


class EarlyStoppingCallback(Callback, HasConfig[EarlyStoppingCallbackConfig]):
    """
    Early Stopping callback monitoring one or multiple metrics.

    This callback stops training if monitored metric(s) do not improve for a
    specified number of evaluation phases (which does not necessarily happen every epoch, see :py:class:`clinicadl.optim.OptimizationConfig`).

    It can monitor multiple metrics simultaneously and allows separate configuration for each metric.
    For any parameter listed below, you may provide either a single value—applied uniformly to all
    monitored metrics—or a sequence of values to configure metrics individually.

    .. note::
        Passing multiple metrics here means that training should stop when **all** the
        monitored metrics have met their stopping criteria. If you want to stop the
        training when **any** of them has met its stopping criterion, you can instantiate
        multiple ``EarlyStoppingCallbacks`` that will monitor each metric independently.

    Parameters
    ----------
    metric : Union[str, Sequence[str]]
        Metric(s) to monitor.
    patience : Union[int, Sequence[int]], default=3
        Number of evaluation phases with no improvement after which training will be stopped.
    min_delta : Union[float, Sequence[float]], default=0.0
        Minimum absolute change in a monitored metric to qualify as an improvement.
    check_finite : Union[bool, Sequence[bool]], default=True
        Whether to stop if the metric becomes NaN or infinite.
    upper_bound : Union[Optional[float], Sequence[Optional[float]]], default=None
        Optional upper threshold that will trigger stopping if exceeded.
    lower_bound : Union[Optional[float], Sequence[Optional[float]]], default=None
        Optional lower threshold that triggers stopping when the value falls below it.
    """

    _config_type = EarlyStoppingCallbackConfig

    def __init__(
        self,
        metric: Union[str, Sequence[str]],
        patience: Union[int, Sequence[int]] = 3,
        min_delta: Union[float, Sequence[float]] = 0.0,
        check_finite: Union[bool, Sequence[bool]] = True,
        upper_bound: Union[Optional[float], Sequence[Optional[float]]] = None,
        lower_bound: Union[Optional[float], Sequence[Optional[float]]] = None,
    ) -> None:
        self.config = self._config_type.from_parameters(
            metric=metric,
            patience=patience,
            min_delta=min_delta,
            check_finite=check_finite,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
        )
        self.stoppers: Optional[list[_OneMetricEarlyStopping]] = None
        self._activated: bool = False  # to prevent from calling in validation only

    def _init_stoppers(self) -> None:
        """
        Initializes all the underlying early stoppers.
        """
        have_modes = all([stopper.mode is not None for stopper in self.config.stoppers])
        if not have_modes:
            raise RuntimeError(
                "Cannot initialize early stoppers because their modes "
                "need to be specified. E.g. by calling on_validation_begin"
            )

        self.stoppers = [config.get_object() for config in self.config.stoppers]

    def _add_modes(self, modes: dict[str, Mode]) -> None:
        """
        Adds their modes to the early stoppers.
        """
        for config in self.config.stoppers:
            config.mode = modes[config.metric]

    def reset(self) -> None:
        """
        To reset metrics monitoring.
        """
        for stopper in self.stoppers:
            stopper.reset()

    # pylint: disable=arguments-differ, unused-argument
    def on_train_begin(self, **kwargs):
        self._activated = True

    def on_validation_begin(self, *, metrics: dict[str, Metric], **kwargs) -> None:
        if self.stoppers is None and self._activated:
            modes = {name: metric.optimum for name, metric in metrics.items()}
            try:
                self._add_modes(modes)
            except KeyError as exc:
                raise get_metric_key_error(exc.args[0]) from exc
            self._init_stoppers()

    def on_validation_end(
        self, *, state: TrainerState, metrics_df: pd.DataFrame, **kwargs
    ) -> None:
        if not self._activated:
            return

        should_stops = [stopper.step(metrics_df, state) for stopper in self.stoppers]
        should_stop = all(should_stops)
        if should_stop:
            logger.info(
                "Early stopping criteria met for all monitored metrics. Stopping training."
            )
        state.should_stop = should_stop

    def state_dict(self) -> Mapping[str, Any]:
        if self.stoppers is None:
            return {}

        return {
            stopper.config.metric: stopper.state_dict() for stopper in self.stoppers
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if self.stoppers is not None and state_dict:
            for stopper in self.stoppers:
                stopper.load_state_dict(state_dict[stopper.config.metric])

    @classmethod
    def _from_config(cls, config):
        args = defaultdict(list)

        for stopper in config.stoppers:
            for k, v in stopper:
                args[k].append(v)

        args.pop("mode")

        early_stopper = cls(**args)

        modes = {stopper.metric: stopper.mode for stopper in config.stoppers}
        early_stopper._add_modes(modes)

        try:
            early_stopper._init_stoppers()
        except RuntimeError:
            pass

        return early_stopper
