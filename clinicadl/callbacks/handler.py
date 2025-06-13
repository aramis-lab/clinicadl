from typing import Dict, List, Optional

from clinicadl.metrics.metrics import ClinicaDLMetrics
from clinicadl.metrics.utils import metric_config_equals
from clinicadl.utils.config.training import _TrainingState

from .factory import *
from .factory.base import Callback


class CallbacksHandler:
    """
    Manages a collection of callback instances to be used during a training pipeline.

    """

    def __init__(
        self,
        callbacks: Optional[List[Callback]] = None,
    ):
        if callbacks is None:
            callbacks = [Chronometer()]

        self.callbacks: Dict[type[Callback], Callback] = {
            type(callback): callback for callback in callbacks
        }

        if Chronometer() not in self.callbacks:
            self.callbacks[Chronometer] = Chronometer()

        if TrainingLoss() not in self.callbacks:
            self.callbacks[TrainingLoss] = TrainingLoss()

        if Logger() not in self.callbacks:
            self.callbacks[Logger] = Logger()

        if ProgressBarCallback() not in self.callbacks:
            self.callbacks[ProgressBarCallback] = ProgressBarCallback()

        for cb in self.callbacks.values():
            if not isinstance(cb, Callback):
                raise TypeError(
                    f"Each custom callback must be a Callback instance, got {type(cb)} for {cb}"
                )

    def check_metrics(self, metrics: ClinicaDLMetrics):
        """TO COMPLETE"""

        if ModelCheckpoint not in self.callbacks.keys():
            self.callbacks[ModelCheckpoint] = ModelCheckpoint(
                metrics=[metrics._loss_metric]
            )

        if EarlyStopping in self.callbacks.keys():
            metrics1 = self.callbacks[EarlyStopping].metrics  # type: ignore
            if not metrics.contains(metrics1):
                metrics.add_metrics(
                    [metric for metric in metrics1 if not metrics.contains([metric])]
                )

            if ModelCheckpoint in self.callbacks.keys():
                metrics2 = self.callbacks[ModelCheckpoint].metrics  # type: ignore
                if not metric_config_equals(metrics1, metrics2):
                    print(metrics1, metrics2)
                    raise ValueError(
                        "EarlyStopping and ModelCheckpoint callbacks must have the same metrics"
                    )
                if not metrics.contains(metrics2):
                    metrics.add_metrics(
                        [metric for metric in metrics2 if metric not in metrics.metrics]
                    )

    @property
    def callback_list(self):
        """
        Get the list of callback class names currently registered.

        Returns
        -------
        list of str
            List of callback class names.
        """
        return [cb.__name__ for cb in self.callbacks.keys()]

    def on_train_begin(self, config: _TrainingState, **kwargs):
        """
        Trigger the `on_train_begin` method of each callback.
        """
        self.call_event("on_train_begin", config=config, **kwargs)

    def on_train_end(self, config: _TrainingState, **kwargs):
        """
        Trigger the `on_train_end` method of each callback.
        """
        self.call_event("on_train_end", config=config, **kwargs)

    def on_epoch_begin(self, config: _TrainingState, **kwargs):
        """
        Trigger the `on_epoch_begin` method of each callback.
        """
        self.call_event("on_epoch_begin", config=config, **kwargs)

    def on_epoch_end(self, config: _TrainingState, **kwargs):
        """
        Trigger the `on_epoch_end` method of each callback.
        """
        self.call_event("on_epoch_end", config=config, **kwargs)

    def on_batch_begin(self, config: _TrainingState, **kwargs):
        """
        Trigger the `on_batch_begin` method of each callback.
        """
        self.call_event("on_batch_begin", config=config, **kwargs)

    def on_batch_end(self, config: _TrainingState, **kwargs):
        """
        Trigger the `on_batch_end` method of each callback.
        """
        self.call_event("on_batch_end", config=config, **kwargs)

    def on_backward_begin(self, config: _TrainingState, **kwargs):
        """
        Trigger the `on_backward_begin` method of each callback.
        """
        self.call_event("on_backward_begin", config=config, **kwargs)

    def on_validation_begin(self, config: _TrainingState, **kwargs):
        """
        Trigger the `on_validation_begin` method of each callback.
        """
        self.call_event("on_validation_begin", config=config, **kwargs)

    def on_validation_end(self, config: _TrainingState, **kwargs):
        """
        Trigger the `on_validation_end` method of each callback.
        """
        self.call_event("on_validation_end", config=config, **kwargs)

    def call_event(self, event, config: _TrainingState, **kwargs):
        """
        Call a specific event method on all callbacks.

        Parameters
        ----------
        event : str
            Name of the event method to call (e.g. 'on_train_begin').

        kwargs : dict
            Keyword arguments passed to each callback's event method.
        """
        for callback in self.callbacks.values():
            method = getattr(callback, event, None)
            if callable(method):
                method(config=config, **kwargs)
