from typing import List, Optional, Union

from .factory import (
    Chronometer,
    CodeCarbonCallback,
    LoggerCallback,
    MLFLOWCallback,
    WandBCallback,
)
from .factory.base import Callback


class CallbacksHandler:
    """
    Manages a collection of callback instances to be used during a training pipeline.

    Built-in support for optional callbacks such as:
    - MLFlow
    - Weights & Biases (WandB)
    - CodeCarbon
    - Logger
    - Chronometer

    Parameters
    ----------
    mlflow : bool, default=False
        Whether to enable MLFLOWCallback.

    wandb : bool, default=False
        Whether to enable WandBCallback.

    codecarbon : bool, default=True
        Whether to enable CodeCarbonCallback.

    logger : bool, default=True
        Whether to enable LoggerCallback.

    chronometer : bool, default=True
        Whether to enable Chronometer.

    custom_callback : Callback or list of Callback, optional
        Custom callback(s) provided by the user.
    """

    def __init__(
        self,
        mlflow: bool = False,
        wandb: bool = False,
        codecarbon: bool = True,
        logger: bool = True,
        chronometer: bool = True,
        custom_callback: Optional[Union[Callback, list[Callback]]] = None,
    ):
        self.callbacks: List[Callback] = []

        if codecarbon:
            self.callbacks.append(CodeCarbonCallback())
        if logger:
            self.callbacks.append(LoggerCallback())
        if chronometer:
            self.callbacks.append(Chronometer())
        if mlflow:
            self.callbacks.append(MLFLOWCallback())
        if wandb:
            self.callbacks.append(WandBCallback())

        if custom_callback is not None:
            if isinstance(custom_callback, list):
                for cb in custom_callback:
                    if not isinstance(cb, Callback):
                        raise TypeError(
                            f"Each custom callback must be a Callback instance, got {type(cb)}"
                        )
                    self.callbacks.append(cb)
            elif isinstance(custom_callback, Callback):
                self.callbacks.append(custom_callback)
            else:
                raise TypeError(
                    f"custom_callback must be a Callback or list of Callback, got {type(custom_callback)}"
                )

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
            self.callbacks.append(callback)

    def remove_callback(self, callback: Callback):
        """
        Remove a single callback to the handler.

        Parameters
        ----------
        callback : Callback
            Callback instance to be added.
        """
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

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
