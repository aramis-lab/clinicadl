"""
    To monitor and control the training process.

    The ``callbacks`` module provides a flexible system for monitoring and controlling
    the training process through modular callback functions.

    Callbacks allow you to hook into different stages of training to perform custom actions
    such as logging, saving checkpoints, or early stopping.

    Some default callbacks are already included in the training loop, but you can also add
    your own by passing them to the :py:class:`~clinicadl.train.trainer.Trainer` via the ``callbacks`` argument:

    .. code-block:: python

        trainer = Trainer(..., callbacks=[MyCustomCallback(), AnotherCallback()])

    Each callback should inherit from the :py:class:`~clinicadl.callbacks.factory.base.Callback` class
    and implement the appropriate event methods, such as: ``on_train_start``, ``on_epoch_end``,
    ``on_validation_end``, etc.

    This makes it easy to customize training behavior without modifying the core training logic.
"""

from .factory import *
