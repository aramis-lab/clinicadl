from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from logging import getLogger
from typing import TYPE_CHECKING, Any, Mapping, Optional, TypeVar, Union

from pydantic import Field, NonNegativeFloat, NonNegativeInt, model_validator
from typing_extensions import Self

from clinicadl.metrics.enum import Optimum
from clinicadl.metrics.handler import MetricsHandler
from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.objects import HasConfig

from ..base import Callback
from .utils import QuantityMonitoring

if TYPE_CHECKING:
    from clinicadl.train import TrainerState


logger = getLogger(__name__)


class OneMetricEarlyStoppingConfig(ObjectConfig["OneMetricEarlyStopping"]):
    """Config class for ``OneMetricEarlyStopping``."""

    metric: str
    patience: NonNegativeInt
    min_delta: NonNegativeFloat
    mode: Optional[Optimum]  # we may not know the mode at first
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
        return OneMetricEarlyStopping


class OneMetricEarlyStopping(
    QuantityMonitoring, HasConfig[OneMetricEarlyStoppingConfig]
):
    """
    Early stopping for a single metric.
    """

    _config_type = OneMetricEarlyStoppingConfig

    def __init__(
        self,
        metric: str,
        patience: int,
        min_delta: float,
        mode: Optimum,
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
        super().__init__(
            name=self.config.metric,
            min_delta=self.config.min_delta,
            mode=self.config.mode,
        )

    # pylint: disable=arguments-renamed
    def step(self, metrics: MetricsHandler, state: TrainerState) -> bool:
        """
        Checks if training should stop.

        Parameters
        ----------
        metrics : MetricsHandler
            The :py:class:`clinicadl.metrics.MetricsHandler` containing the validation metrics.
        state : TrainerState
            The state of the trainer.

        Returns
        -------
        bool
            The decision.
        """
        value = metrics.get_metric_value(
            metric=self.config.metric, epoch=state.current_epoch
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

        super().step(value, log=True)

        if self.num_non_improvements > self.config.patience:
            logger.info(
                "Early stopping triggered on metric '%s' after %s evaluation(s) without improvement.",
                self.config.metric,
                self.num_non_improvements,
            )
            return True

        return False


T = TypeVar("T")


class EarlyStoppingCallbackConfig(ObjectConfig["EarlyStoppingCallback"]):
    """Config class for ``EarlyStoppingCallback``."""

    stoppers: Sequence[OneMetricEarlyStoppingConfig] = Field(
        reader=lambda stoppers: list(
            map(OneMetricEarlyStoppingConfig.from_dict, stoppers)
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
                OneMetricEarlyStoppingConfig(
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

    .. note::
        No need to specify if the monitored quantity should be minimized or maximized,
        it is specified in :py:attr:`clinicadl.metric.Metric.optimum`.

    Parameters
    ----------
    metric : Union[str, Sequence[str]]
        Metric(s) to monitor.
    patience : Union[int, Sequence[int]], default=3
        The number of allowed evaluation phases with no improvement. For example, if ``patience=0``
        the stop signal is triggered as soon as the monitored quantity is not improved.
    min_delta : Union[float, Sequence[float]], default=0.0
        Minimum absolute change in a monitored metric to qualify as an improvement.
    check_finite : Union[bool, Sequence[bool]], default=True
        Whether to stop if the metric becomes NaN or infinite.
    upper_bound : Union[Optional[float], Sequence[Optional[float]]], default=None
        Optional upper threshold that will trigger stopping if exceeded.
    lower_bound : Union[Optional[float], Sequence[Optional[float]]], default=None
        Optional lower threshold that triggers stopping when the value falls below it.

    Examples
    --------
    .. code-block::
        from clinicadl.callbacks import EarlyStoppingCallback
        from clinicadl.train import Trainer
        from clinicadl.metrics.config import MSEMetricConfig, LossMetricConfig
        ...

        trainer = Trainer(
            metrics={"loss": LossMetricConfig(), "mse": MSEMetricConfig()},
            callbacks=[EarlyStoppingCallback(metric="mse", patience=5)],
            ...
        )
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
        self.stoppers: Optional[list[OneMetricEarlyStopping]] = None

    def _init_stoppers(self) -> None:
        """
        Initializes all the underlying early stoppers.
        """
        have_modes = all(stopper.mode is not None for stopper in self.config.stoppers)
        if not have_modes:
            raise RuntimeError(
                "Cannot initialize early stoppers because their modes "
                "need to be specified. E.g. by calling on_train_start"
            )

        self.stoppers = [config.get_object() for config in self.config.stoppers]

    def _add_modes(self, modes: dict[str, Optimum]) -> None:
        """
        Adds their modes to the early stoppers.
        """
        for config in self.config.stoppers:
            config.mode = modes[config.metric]

    def _reset(self) -> None:
        """
        To reset metrics monitoring.
        """
        if self.stoppers:
            for stopper in self.stoppers:
                stopper.reset()

    # pylint: disable=arguments-differ, unused-argument
    def on_train_start(self, metrics: MetricsHandler, **kwargs):
        self._reset()

        for config in self.config.stoppers:
            metrics.check_metric_name(metric=config.metric)
        modes = {name: metric.optimum for name, metric in metrics.metrics.items()}
        self._add_modes(modes)
        self._init_stoppers()

    def on_resume(self, **kwargs):
        self._init_stoppers()

    def on_validation_end(
        self, *, state: TrainerState, metrics: MetricsHandler, **kwargs
    ) -> None:
        should_stops = [stopper.step(metrics, state) for stopper in self.stoppers]
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
