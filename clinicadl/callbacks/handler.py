from collections import Counter
from copy import copy
from typing import Sequence, TypeVar

from pydantic import Field, field_validator

from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.objects import HasConfig

from .base import Callback, Event
from .factory import get_callback_from_dict
from .implemented import (
    ChecksCallback,
    ConfigSaverCallback,
    LoggerCallback,
    MetricsSaverCallback,
    ModelCheckpointCallback,
    MonitorCallback,
    TrainingCheckpointCallback,
    TrainingLossCallback,
)

T = TypeVar("T")

FIRST_LAST = [LoggerCallback]
FIRST = [
    ChecksCallback,
    ConfigSaverCallback,
    MonitorCallback,
]
LAST = [TrainingCheckpointCallback]
ONLY_ONE = [
    LoggerCallback,
    ChecksCallback,
    ConfigSaverCallback,
    MonitorCallback,
    TrainingLossCallback,
    TrainingCheckpointCallback,
]


class CallbacksHandlerConfig(ObjectConfig["CallbacksHandler"]):
    """
    To check the callbacks passed by the user.
    """

    callbacks: list[Callback] = Field(
        reader=lambda callbacks: [get_callback_from_dict(c) for c in callbacks]
    )

    @field_validator("callbacks", mode="after")
    @classmethod
    def _check_duplicate(cls, callbacks: list[Callback]) -> list[Callback]:
        """
        Checks unwanted duplicates.
        """
        counts = Counter(type(callback) for callback in callbacks)
        for callback in ONLY_ONE:
            if counts.get(callback, 0) > 1:
                raise ValueError(f"You cannot pass more than one {callback}")

        return callbacks

    def add_callbacks(
        self,
        callbacks: Sequence[Callback],
    ) -> None:
        """
        Adds new callbacks.

        Parameters
        ----------
        callbacks : Sequence[Callback]
            The :py:class:`Callbacks <clinicadl.callbacks.Callback` to add.
        """
        self.callbacks += callbacks

    @classmethod
    def _get_class(cls):
        return CallbacksHandlerConfig


def _reorder(
    input_list: list[T], order: list[type], unordered_at_the_end: bool
) -> None:
    """
    Reorders elements according to their types and the order given in 'order'. If unordered_at_the_end=True, all the elements whose
    types are not in ordered will stay in the same order but at the end; otherwise, they will stay in the same order but at
    the beginning.
    """
    if unordered_at_the_end:
        default_rank = float("inf")
    else:
        default_rank = -float("inf")
    rank = {cls_: i for i, cls_ in enumerate(order)}
    input_list.sort(key=lambda x: rank.get(type(x), default_rank))


class CallbacksHandler(HasConfig[CallbacksHandlerConfig]):
    """
    To handle all the :py:class:`Callbacks <clinicadl.callbacks.Callback>` passed to a :py:class:`~clinicadl.train.Trainer`.

    Note that some callbacks are instantiated by default: :py:class:`~clinicadl.callbacks.LoggerCallback`,
    :py:class:`~clinicadl.callbacks.MonitorCallback`, :py:class:`~clinicadl.callbacks.ModelCheckpointCallback`
    and :py:class:`~clinicadl.callbacks.TrainingCheckpointCallback`. To override them, just pass a new instance.

    Parameters
    ----------
    callbacks : Sequence[Callback]
        A sequence of :py:class:`Callbacks <clinicadl.callbacks.Callback>`.

        .. important:: Order matters!
            The order of the callbacks may determine the order in which callbacks will be called.
            Note, however, that some callbacks have an immutable rank in this order. For example, no matter where you place
            :py:class:`~clinicadl.callbacks.LoggerCallback`, it will be called first to initialize logging and last to
            shutdown logging.
    """

    _config_type = CallbacksHandlerConfig

    def __init__(
        self,
        callbacks: Sequence[Callback],
    ):
        self.config = self._config_type(callbacks=callbacks)
        self._all = None  # inputs + defaults + mandatory
        self._with_defaults = None  # inputs + defaults
        self._ordered = None  # inputs + defaults + mandatory, except the ones in self._first_and_last, ordered
        self._first_and_last = None  # callbacks that are always called first and last
        self._complete_callbacks()

    @property
    def callbacks(self) -> list[Callback]:
        """The public callbacks currently in the ``CallbacksHandler``."""
        return self._with_defaults

    @property
    def all_callbacks(self) -> list[Callback]:
        """
        The callbacks currently in the ``CallbacksHandler``, including private callbacks
        (mandatory callbacks that are always used by ``ClinicaDL``).
        """
        return self._all

    def add_callbacks(
        self,
        callbacks: Sequence[Callback],
    ) -> None:
        """
        Adds new callbacks.
        """
        self.config.add_callbacks(callbacks)
        self._complete_callbacks()

    def call_event(self, event: str | Event, **kwargs) -> None:
        """
        Call a specific event method on all callbacks
        (see :py:class:`~clinicadl.callbacks.Callback` to get
        the list of the events).

        Parameters
        ----------
        event : str | Event
            Name of the event to call (e.g. ``"on_train_start"``).
        kwargs : Any
            Keyword arguments that will be passed to the methods
            associated to this event.
        """
        event = Event(event).value

        is_start_event = (
            event.endswith("start") or event.endswith("init") or event == Event.RESUME
        )
        is_end_event = event.endswith("end") or event == Event.EXCEPTION
        assert is_start_event or is_end_event

        if is_start_event:
            for callback in self._first_and_last:
                getattr(callback, event)(**kwargs)

        for callback in self._ordered:
            getattr(callback, event)(**kwargs)

        if is_end_event:
            for callback in self._first_and_last[::-1]:
                getattr(callback, event)(**kwargs)

    def _complete_callbacks(self) -> None:
        """
        Completes the callbacks passed by the user with the mandatory
        and default callbacks, and orders the callbacks.
        """
        callbacks = copy(self.config.callbacks)
        self._add_defaults(callbacks)
        self._reorder(callbacks)
        self._with_defaults = copy(callbacks)
        self._add_mandatory(callbacks)
        self._all = copy(callbacks)
        self._reorder(callbacks)
        self._ordered = [
            callback for callback in callbacks if type(callback) not in FIRST_LAST
        ]
        self._first_and_last = [
            callback for callback in callbacks if type(callback) in FIRST_LAST
        ]

    @classmethod
    def _add_mandatory(cls, callbacks: list[Callback]) -> None:
        """
        Adds the mandatory callbacks.
        """
        callbacks.extend(cls._get_mandatory())

    @classmethod
    def _add_defaults(cls, callbacks: list[Callback]) -> None:
        """
        Adds the default callbacks.
        """
        for callback in cls._get_default():
            if not any(type(x) is type(callback) for x in callbacks):
                callbacks.append(callback)

    @staticmethod
    def _reorder(callbacks: list[Callback]) -> None:
        """
        Reorders the callbacks.
        """
        _reorder(callbacks, FIRST, unordered_at_the_end=True)
        _reorder(callbacks, FIRST_LAST, unordered_at_the_end=True)
        _reorder(callbacks, LAST, unordered_at_the_end=False)

    @staticmethod
    def _get_default() -> list[Callback]:
        return [
            LoggerCallback(),
            MonitorCallback(),
            ModelCheckpointCallback(save_last=True),
            TrainingCheckpointCallback(),
        ]

    @staticmethod
    def _get_mandatory() -> list[Callback]:
        return [
            ChecksCallback(),
            ConfigSaverCallback(),
            TrainingLossCallback(),
            MetricsSaverCallback(),
        ]
