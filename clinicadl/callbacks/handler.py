from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pydantic import Field

from clinicadl.io import Maps
from clinicadl.metrics.handler import MetricsHandler
from clinicadl.models import Model
from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.json import read_json, write_json
from clinicadl.utils.objects import HasConfig
from clinicadl.utils.typing import PathType

from .base import Callback, Events
from .factory import get_callback_from_dict
from .implemented import *

# LOSS = "loss"

# PREFERRED_ORDER = [
#     _TrainingLoss.__name__,
#     LRScheduler.__name__,
#     Checkpoint.__name__,
#     ModelSelection.__name__,
#     _Monitor.__name__,
#     _Logger.__name__,
#     MLflow.__name__,
#     CodeCarbon.__name__,
#     WandB.__name__,
#     TensorBoard.__name__,
# ]


class CallbacksHandlerConfig(ObjectConfig["CallbacksHandler"]):
    """
    To check the callbacks passed by the user.
    """

    callbacks: list[Callback] = Field(
        reader=lambda callbacks: [get_callback_from_dict(c) for c in callbacks]
    )

    def add_callbacks(
        self,
        callbacks: Sequence[Callback],
    ) -> None:
        """
        Adds new callbacks.
        """
        self.callbacks += callbacks

    @classmethod
    def _get_class(cls):
        return CallbacksHandlerConfig


class CallbacksHandler(HasConfig[CallbacksHandlerConfig]):
    """
    Central handler for all training callbacks in the ClinicaDL pipeline.

    This class initializes, validates, and orders the various callbacks used during
    training and evaluation, including logging, monitoring, early stopping, and checkpointing.

    Parameters
    ----------
    metrics : MetricsHandler
        Object used to validate the metric names required by `ModelSelection` and `EarlyStopping`.
    callbacks : Optional[List[Callback]]
        List of user-defined callbacks. Supports duplicates for EarlyStopping,
        and merges `ModelSelection` instances.

    Attributes
    ----------
    callbacks : Dict[str, Callback]
        Dictionary mapping callback identifiers to instantiated callbacks.

    See Also
    --------
    :py:class:`~clinicadl.callbacks.base.Callback`: Abstract base class for all callbacks.
    :py:class:`~clinicadl.metrics.metrics.MetricsHandler`: Metric management utility.
    """

    _config_type = CallbacksHandlerConfig

    def __init__(
        self,
        callbacks: Sequence[Callback],
    ):
        self.config = self._config_type(callbacks=callbacks)

    @property
    def callbacks(self) -> list[Callback]:
        """The metrics currently in the MetricsHandler."""
        return self.config.callbacks

    def add_callbacks(
        self,
        callbacks: Sequence[Callback],
    ) -> None:
        """
        Adds new callbacks.
        """
        self.config.add_callbacks(callbacks)

    def call_event(self, event: Events, **kwargs) -> None:
        """
        Call a specific event method on all callbacks.

        Parameters
        ----------
        event : str
            Name of the event method to call (e.g. 'on_train_start').

        kwargs : dict
            Keyword arguments passed to each callback's event method.
        """
        for callback in self.callbacks:
            getattr(callback, Events(event).value)(**kwargs)

    # def _check_callbacks_names(
    #     self, callbacks: Optional[List[Callback]] = None
    # ) -> Dict[str, Callback]:
    #     """
    #     Resolve user-provided callbacks, avoiding duplicates and handling special cases.

    #     Parameters
    #     ----------
    #     callbacks : Optional[List[Callback]]
    #         List of callback instances.

    #     Returns
    #     -------
    #     Dict[str, Callback]
    #         Mapping of callback names to instances.
    #     """
    #     if not callbacks:
    #         return {}

    #     resolved: Dict[str, Callback] = {}

    #     for callback in callbacks:
    #         if not isinstance(callback, Callback):
    #             raise TypeError(
    #                 f"Each callback must be a Callback instance, got {type(callback)} for {callback}"
    #             )

    #         name = type(callback).__name__

    #         if isinstance(callback, (EarlyStopping, LRScheduler)):
    #             count = sum(k.startswith(name) for k in resolved)
    #             unique_name = f"{name}{count + 1}" if name in resolved else name
    #             resolved[unique_name] = callback

    #         elif isinstance(callback, ModelSelection):
    #             if name in resolved:
    #                 merged_metrics = set(resolved[name].metrics).union(callback.metrics)  # type: ignore
    #                 resolved[name] = ModelSelection(metrics=list(merged_metrics))
    #             else:
    #                 resolved[name] = callback

    #         elif name in resolved:
    #             raise ValueError(f"Duplicate callback not allowed: {name}")

    #         else:
    #             resolved[name] = callback

    #     return resolved

    # def _add_default_callbacks(self):
    #     """
    #     Add default callbacks if they are not already provided.
    #     """
    #     defaults = {
    #         _Monitor.__name__: _Monitor(),
    #         _TrainingLoss.__name__: _TrainingLoss(),
    #         _Logger.__name__: _Logger(),
    #         _CheckpointSaver.__name__: _CheckpointSaver(),
    #     }

    #     for name, callback in defaults.items():
    #         if name not in self.callbacks:
    #             self.callbacks[name] = callback

    # def _check_metrics(self, metrics: MetricsHandler):
    #     """
    #     Ensure that all metrics used in ModelSelection and EarlyStopping callbacks
    #     are present in the provided metrics.

    #     Raises
    #     ------
    #     ValueError
    #         If any metric required by `EarlyStopping` or `ModelSelection` is missing.
    #     """

    #     es_metrics = [
    #         metric
    #         for cb in self.callbacks.values()
    #         if isinstance(cb, EarlyStopping)
    #         for metric in cb.metrics
    #     ]

    #     ms_cb = self.callbacks.get(ModelSelection.__name__)
    #     ms_metrics = set(ms_cb.metrics if ms_cb else [])  # type: ignore
    #     ms_metrics.add(LOSS)

    #     ms_metrics.update(es_metrics)
    #     self.callbacks[ModelSelection.__name__] = ModelSelection(
    #         metrics=list(ms_metrics)
    #     )

    #     available = set(metrics.metrics.keys())

    #     if es_metrics:
    #         missing_early = set(es_metrics) - available
    #         if missing_early:
    #             raise ValueError(f"Missing metrics for EarlyStopping: {missing_early}")

    #     missing = ms_metrics - available
    #     if missing:
    #         raise ValueError(f"Missing metrics for ModelSelection: {missing}")

    # def _check_callbacks_order(self):
    #     """
    #     Order callbacks based on preferred priority.
    #     """

    #     early = {
    #         k: v
    #         for k, v in self.callbacks.items()
    #         if k.startswith(EarlyStopping.__name__)
    #     }
    #     rest = {k: v for k, v in self.callbacks.items() if k not in early}

    #     ordered = dict(sorted(early.items()))
    #     ordered.update(
    #         {name: rest.pop(name) for name in PREFERRED_ORDER if name in rest}
    #     )
    #     ordered.update(rest)

    #     self.callbacks = ordered

    # @property
    # def callback_list(self) -> list[str]:
    #     """
    #     Get the list of callback class names currently registered.

    #     Returns
    #     -------
    #     list of str
    #         List of callback class names.
    #     """
    #     return list(self.callbacks.keys())
