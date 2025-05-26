from typing import Dict, List, Optional, Union

from clinicadl.maps.maps import Maps
from clinicadl.metrics.utils import metric_config_equals
from clinicadl.model.clinicadl_model import ClinicaDLModel

from .factory import *
from .factory.base import Callback


class CallbacksHandler:
    """
    Manages a collection of callback instances to be used during a training pipeline.

    """

    def __init__(
        self,
        maps: Maps,
        model: ClinicaDLModel,
        callbacks: Optional[List[Callback]] = None,
    ):
        if callbacks is None:
            callbacks = []

        self.callbacks: Dict[type[Callback], Callback] = {
            type(callback): callback for callback in callbacks
        }

        self.add_callback(Chronometer())

        self.maps = maps
        self.model = model

        for cb in self.callbacks:
            if not isinstance(cb, Callback):
                raise TypeError(
                    f"Each custom callback must be a Callback instance, got {type(cb)}"
                )

        if EarlyStopping in self.callbacks.keys():
            if ModelCheckpoint in self.callbacks.keys():
                metrics1 = self.callbacks[EarlyStopping].metrics
                metrics2 = self.callbacks[ModelCheckpoint].metrics
                if not metric_config_equals(metrics1, metrics2):
                    raise ValueError(
                        "EarlyStopping and ModelCheckpoint callbacks must have the same metrics"
                    )

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                f"You are adding a {cb_class} to the callbacks of this Trainer, but there is already one. The current"
                + "list of callbacks is\n:"
                + self.callback_list
            )
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    def add_callback(self, callback: Callback):
        """
        Add a single callback to the handler.

        Parameters
        ----------
        callback : Callback
            Callback instance to be added.
        """
        if not isinstance(callback, Callback):
            raise TypeError(
                f"callback must be an instance of Callback, got {type(callback)}"
            )

        if callback not in self.callbacks:
            self.callbacks[type(callback)] = callback

    @property
    def callback_list(self):
        """
        Get the list of callback class names currently registered.

        Returns
        -------
        list of str
            List of callback class names.
        """
        return [cb.__class__.__name__ for cb in self.callbacks]

    def on_train_begin(self, **kwargs):
        """
        Trigger the `on_train_begin` method of each callback.
        """
        self.call_event("on_train_begin", **kwargs)

    def on_train_end(self, **kwargs):
        """
        Trigger the `on_train_end` method of each callback.
        """
        self.call_event("on_train_end", **kwargs)

    def on_epoch_begin(self, **kwargs):
        """
        Trigger the `on_epoch_begin` method of each callback.
        """
        self.call_event("on_epoch_begin", **kwargs)

    def on_epoch_end(self, **kwargs):
        """
        Trigger the `on_epoch_end` method of each callback.
        """
        self.call_event("on_epoch_end", **kwargs)

    def on_batch_begin(self, **kwargs):
        """
        Trigger the `on_batch_begin` method of each callback.
        """
        self.call_event("on_batch_begin", **kwargs)

    def on_batch_end(self, **kwargs):
        """
        Trigger the `on_batch_end` method of each callback.
        """
        self.call_event("on_batch_end", **kwargs)

    def on_backward_begin(self, **kwargs):
        """
        Trigger the `on_backward_begin` method of each callback.
        """
        self.call_event("on_backward_begin", **kwargs)

    def on_validation_begin(self, **kwargs):
        """
        Trigger the `on_validation_begin` method of each callback.
        """
        self.call_event("on_validation_begin", **kwargs)

    def on_validation_end(self, **kwargs):
        """
        Trigger the `on_validation_end` method of each callback.
        """
        self.call_event("on_validation_end", **kwargs)

    def call_event(self, event, **kwargs):
        """
        Call a specific event method on all callbacks.

        Parameters
        ----------
        event : str
            Name of the event method to call (e.g. 'on_train_begin').

        kwargs : dict
            Keyword arguments passed to each callback's event method.
        """
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if callable(method):
                method(**kwargs)


# TODO: add WandB, MLFLOW, CodeCarbon, Tensorboard, LearningRateScheduler, EarlyStopping, ModelCheckpoint etc...
