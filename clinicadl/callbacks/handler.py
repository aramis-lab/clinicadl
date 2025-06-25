from typing import Dict, List, Optional

from clinicadl.metrics.metrics import ClinicaDLMetrics
from clinicadl.train.training_state import _TrainingState

from .factory import *
from .factory.base import Callback

LOSS = "loss"


class CallbacksHandler:
    """
    Class that manages a collection of callbacks to be executed during the
    different stages of a training pipeline in ClinicaDL.

    This handler centralizes all callback logic (e.g., early stopping, logging,
    metric selection, time tracking) and ensures that each registered callback
    is executed at appropriate times during training and validation.

    .. note::
        Custom callbacks can be passed during initialization. Additionally,
        default callbacks such as :ref:`Chronometer`, :ref:`TrainingLoss`, :ref:`Logger`,
        :ref:`ProgressBar`, and :ref:`CurrentState` are added automatically if not already provided.

    Parameters
    ----------
    callbacks : list of Callback, optional
        List of custom callback instances to use. Each callback must inherit
        from `clinicadl.utils.callbacks.factory.base.Callback`.

    Attributes
    ----------
    callbacks : dict
        A dictionary mapping callback class names to instances.
        Each callback is uniquely identified by its class name, unless duplicates are allowed
        (e.g., multiple `EarlyStopping` callbacks).

    Methods
    -------
    check_metrics(metrics: ClinicaDLMetrics)
        Ensure that all metrics required by ModelSelection and EarlyStopping are available.

    on_train_begin(config: _TrainingState, **kwargs)
        Triggered at the start of training.

    on_train_end(config: _TrainingState, **kwargs)
        Triggered at the end of training.

    on_epoch_begin(config: _TrainingState, **kwargs)
        Triggered at the start of each epoch.

    on_epoch_end(config: _TrainingState, **kwargs)
        Triggered at the end of each epoch.

    on_batch_begin(config: _TrainingState, **kwargs)
        Triggered before each training batch is processed.

    on_batch_end(config: _TrainingState, **kwargs)
        Triggered after each training batch is processed.

    on_backward_begin(config: _TrainingState, **kwargs)
        Triggered before backward pass (if needed).

    on_validation_begin(config: _TrainingState, **kwargs)
        Triggered before starting validation loop.

    on_validation_end(config: _TrainingState, **kwargs)
        Triggered after ending validation loop.

    callback_list : list of str
        Property returning the list of all currently registered callback names.

    Examples
    --------
    >>> from clinicadl.utils.callbacks.handler import CallbacksHandler
    >>> from clinicadl.utils.callbacks.factory import EarlyStopping
    >>> handler = CallbacksHandler(callbacks=[EarlyStopping(patience=3)])
    >>> handler.on_train_begin(config)

    Notes
    -----
    - When multiple EarlyStopping callbacks are passed, they are renamed to avoid collisions.
    - If EarlyStopping and ModelSelection use different metrics, the union of their metrics is used in ModelSelection.
    - LOSS is always enforced in ModelSelection to ensure compatibility with training monitoring.

    See Also
    --------
    clinicadl.utils.callbacks.factory.base.Callback
    clinicadl.metrics.metrics.ClinicaDLMetrics
    """

    def __init__(
        self,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.callbacks: Dict[str, Callback] = {}
        if callbacks:
            for callback in callbacks:
                if not isinstance(callback, Callback):
                    raise TypeError(
                        f"Each custom callback must be a Callback instance, got {type(callback)} for {callback}"
                    )

                callback_name = callback.__class__.__name__

                if callback_name not in self.callbacks:
                    self.callbacks[callback_name] = callback

                elif isinstance(callback, EarlyStopping):
                    callback_name = callback_name + str(
                        sum(callback_name in key for key in self.callbacks) + 1
                    )
                    self.callbacks[callback_name] = callback

                elif isinstance(callback, ModelSelection):
                    existing = set(self.callbacks[callback_name].metrics)
                    new = set(callback.metrics)
                    self.callbacks[callback_name] = ModelSelection(
                        metrics=list(existing.union(new))
                    )

                else:
                    raise ValueError(
                        f"Callback {callback_name} is already registered, you can't provide it twice."
                    )

        # Default callbacks

        if Chronometer.__name__ not in self.callbacks:
            self.callbacks[Chronometer.__name__] = Chronometer()

        if TrainingLoss.__name__ not in self.callbacks:
            self.callbacks[TrainingLoss.__name__] = TrainingLoss()

        if Logger.__name__ not in self.callbacks:
            self.callbacks[Logger.__name__] = Logger()

        if CurrentState.__name__ not in self.callbacks:
            self.callbacks[CurrentState.__name__] = CurrentState()

    def check_metrics(self, metrics: ClinicaDLMetrics):
        """
        Ensure that all metrics used in ModelSelection and EarlyStopping callbacks
        are present in the provided metrics.

        Raises
        ------
        ValueError
            If any required metric is missing from `metrics`.
        """

        early_stopping_metrics = []

        if ModelSelection.__name__ not in self.callbacks:
            if EarlyStopping.__name__ not in self.callbacks:
                self.callbacks[ModelSelection.__name__] = ModelSelection(metrics=[LOSS])

                model_selection_metrics = [LOSS]

            else:
                # TODO : check if loss not in EarlyStopping, do I add it to ModelSelection ?
                for _, v in self.callbacks.items():
                    if isinstance(v, EarlyStopping):
                        early_stopping_metrics.extend(v.metrics)

                self.callbacks[ModelSelection.__name__] = ModelSelection(
                    metrics=early_stopping_metrics
                )

                model_selection_metrics = early_stopping_metrics
        else:
            model_selection_metrics: list[str] = self.callbacks[
                ModelSelection.__name__
            ].metrics  # type: ignore

            if LOSS not in model_selection_metrics:
                model_selection_metrics.append(LOSS)
                self.callbacks[ModelSelection.__name__] = ModelSelection(
                    metrics=model_selection_metrics
                )

            if EarlyStopping.__name__ in self.callbacks:
                for _, v in self.callbacks.items():
                    if isinstance(v, EarlyStopping):
                        early_stopping_metrics.extend(v.metrics)

                diff = list(set(early_stopping_metrics) - set(model_selection_metrics))

                if len(diff) > 0:
                    model_selection_metrics.extend(diff)
                    self.callbacks[ModelSelection.__name__] = ModelSelection(
                        metrics=model_selection_metrics
                    )

        available_metrics = metrics.metrics.keys()

        if not all(elem in available_metrics for elem in model_selection_metrics):
            raise ValueError(
                f"Some metrics from ModelSelection are not in the metrics: \n"
                f"{[x for x in model_selection_metrics if x not in available_metrics]}"
            )
        if not all(elem in available_metrics for elem in early_stopping_metrics):
            raise ValueError(
                f"Some metrics from EarlyStopping are not in the metrics: \n"
                f"{[x for x in early_stopping_metrics if x not in available_metrics]}"
            )

    @property
    def callback_list(self) -> list[str]:
        """
        Get the list of callback class names currently registered.

        Returns
        -------
        list of str
            List of callback class names.
        """
        return list(self.callbacks.keys())

    def on_train_begin(self, config: _TrainingState, **kwargs):
        """
        Trigger the `on_train_begin` method of each callback.
        """
        self._call_event("on_train_begin", config=config, **kwargs)

    def on_train_end(self, config: _TrainingState, **kwargs):
        """
        Trigger the `on_train_end` method of each callback.
        """
        self._call_event("on_train_end", config=config, **kwargs)

    def on_epoch_begin(self, config: _TrainingState, **kwargs):
        """
        Trigger the `on_epoch_begin` method of each callback.
        """
        self._call_event("on_epoch_begin", config=config, **kwargs)

    def on_epoch_end(self, config: _TrainingState, **kwargs):
        """
        Trigger the `on_epoch_end` method of each callback.
        """
        self._call_event("on_epoch_end", config=config, **kwargs)

    def on_batch_begin(self, config: _TrainingState, **kwargs):
        """
        Trigger the `on_batch_begin` method of each callback.
        """
        self._call_event("on_batch_begin", config=config, **kwargs)

    def on_batch_end(self, config: _TrainingState, **kwargs):
        """
        Trigger the `on_batch_end` method of each callback.
        """
        self._call_event("on_batch_end", config=config, **kwargs)

    def on_backward_begin(self, config: _TrainingState, **kwargs):
        """
        Trigger the `on_backward_begin` method of each callback.
        """
        self._call_event("on_backward_begin", config=config, **kwargs)

    def on_validation_begin(self, config: _TrainingState, **kwargs):
        """
        Trigger the `on_validation_begin` method of each callback.
        """
        self._call_event("on_validation_begin", config=config, **kwargs)

    def on_validation_end(self, config: _TrainingState, **kwargs):
        """
        Trigger the `on_validation_end` method of each callback.
        """
        self._call_event("on_validation_end", config=config, **kwargs)

    def _call_event(self, event, config: _TrainingState, **kwargs):
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
