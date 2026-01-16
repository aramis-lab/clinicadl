from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Optional

import pandas as pd
import torch

if TYPE_CHECKING:
    from clinicadl.data.dataloader import Batch, BatchType, DataLoader
    from clinicadl.io import Maps
    from clinicadl.losses.types import LossType
    from clinicadl.metrics import MetricsHandler
    from clinicadl.models import Model
    from clinicadl.optim.config import OptimizationConfig
    from clinicadl.split import Split
    from clinicadl.train import TrainerState
    from clinicadl.train.computational import ComputationalConfig

    from .handler import CallbacksHandler


class Events(str, Enum):
    """Events that can trigger an action from a :py:class:`clinicadl.callbacks.Callback`."""

    EXCEPTION = "on_exception"
    INIT = "on_trainer_init"

    # Training
    TRAIN_START = "on_train_start"
    TRAIN_END = "on_train_end"
    EPOCH_START = "on_epoch_start"
    EPOCH_END = "on_epoch_end"
    BATCH_START = "on_batch_start"
    BATCH_END = "on_batch_end"
    FORWARD_START = "on_forward_step_start"
    BACKWARD_START = "on_backward_step_start"
    BACKWARD_END = "on_backward_step_end"
    OPTIM_STEP_START = "on_optimization_step_start"
    OPTIM_STEP_END = "on_optimization_step_end"

    # Validation
    VAL_START = "on_validation_start"
    VAL_END = "on_validation_end"
    EVAL_START = "on_evaluation_step_start"
    EVAL_END = "on_evaluation_step_end"

    # Test
    TEST_START = "on_test_start"
    TEST_END = "on_test_end"

    # Predict
    PREDICT_START = "on_predict_start"
    PREDICT_END = "on_predict_end"


class Callback(ABC):
    """
    To define arbitrary action to perform at certain points of the training, evaluation or
    prediction loop.

    Each method of this class starting by ``on_...`` is associated to an event of
    the training, evaluation or prediction phase of the :py:class:`~clinicadl.train.Trainer`.
    By overriding these methods, the user can define action to perform when the event happens.

    .. important::
        Callbacks should capture NON-ESSENTIAL logic such as saving checkpoints or logging.
        The essential logic should be defined in a :py:class:`clinicadl.models.Model`.

    To define you own callback, you can override any of the method associated to an event,
    and you must override :py:meth:`state_dict` and :py:meth:`load_state_dict`.

    """

    def reset(self) -> None:
        """
        Called every time :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`, :py:meth:`Trainer.validate <clinicadl.train.Trainer.validate>`,
        :py:meth:`Trainer.test <clinicadl.train.Trainer.test>` or :py:meth:`Trainer.predict <clinicadl.train.Trainer.predict>` are called.
        """

    def on_trainer_init(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        metrics: MetricsHandler,
        optimization: OptimizationConfig,
        callbacks: CallbacksHandler,
    ) -> None:
        """
        Called once when the :py:class:`~clinicadl.train.Trainer` is instantiated.

        Parameters
        ----------
        model : Model
            The :py:class:`clinicadl.models.Model` associated to the :py:class:`clinicadl.train.Trainer`.
        maps : Maps
            The :py:class:`clinicadl.io.Maps` associated to the :py:class:`clinicadl.train.Trainer`.
        state : TrainerState
            The current :py:class:`clinicadl.train.TrainerState`.
        metrics : MetricsHandler
            The :py:class:`~clinicadl.metrics.MetricsHandler` containing the metrics passed to the :py:class:`~clinicadl.train.Trainer`.
        optimization : OptimizationConfig
            The :py:class:`clinicadl.optim.OptimizationConfig` defining the optimization specifications
            of the training phase.
        callbacks : CallbacksHandler
            The :py:class:`~clinicadl.callbacks.CallbacksHandler` containing the callbacks passed to the :py:class:`~clinicadl.train.Trainer`.
        """

    def on_exception(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        exception: Exception,
    ) -> None:
        """
        Called when an exception interrupts an execution of the :py:class:`~clinicadl.train.Trainer`.

        Parameters
        ----------
        model : Model
            The :py:class:`clinicadl.models.Model` associated to the :py:class:`~clinicadl.train.Trainer`.
        maps : Maps
            The :py:class:`clinicadl.io.Maps` associated to the :py:class:`clinicadl.train.Trainer`.
        state : TrainerState
            The current :py:class:`clinicadl.train.TrainerState`.
        exception : Exception
            The exception that has been raised.
        """

    # Train

    def on_train_start(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        split: Split,
        optimizers: dict[str, torch.optim.Optimizer],
        optimization: OptimizationConfig,
        computational: ComputationalConfig,
    ) -> None:
        """
        Called once at the beginning of :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`.

        Parameters
        ----------
        model : Model
            The :py:class:`clinicadl.models.Model` associated to the :py:class:`clinicadl.train.Trainer`.
        maps : Maps
            The :py:class:`clinicadl.io.Maps` associated to the :py:class:`clinicadl.train.Trainer`.
        state : TrainerState
            The current :py:class:`clinicadl.train.TrainerState`.
        split : Split
            The :py:class:`clinicadl.split.Split` on which training is performed.
        optimizers : dict[str, torch.optim.Optimizer]
            The :py:class`Optimizer <torch.optim.Optimizer>` returned
            by :py:meth:`Model.backward_step <clinicadl.models.Model.build_optimizers>`.
        optimization : OptimizationConfig
            The :py:class:`clinicadl.optim.OptimizationConfig` defining the optimization specifications
            of the training phase.
        computational : ComputationalConfig
            The :py:class:`clinicadl.train.ComputationalConfig` defining the computational specifications
            of the training phase.
        callbacks : list[Callback]
            The list of :py:class:`Callbacks <clinicadl.callbacks.Callback>` associated to the :py:class:`~clinicadl.train.Trainer`.
        """

    def on_train_end(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
    ) -> None:
        """
        Called once at the end of :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`.

        Parameters
        ----------
        model : Model
            The :py:class:`clinicadl.models.Model` associated to the :py:class:`clinicadl.train.Trainer`.
        maps : Maps
            The :py:class:`clinicadl.io.Maps` associated to the :py:class:`clinicadl.train.Trainer`.
        state : TrainerState
            The current :py:class:`clinicadl.train.TrainerState`.
        """

    def on_epoch_start(self, *, model: Model, maps: Maps, state: TrainerState) -> None:
        """
        Called at the beginning of an epoch in :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`.

        Parameters
        ----------
        model : Model
            The :py:class:`clinicadl.models.Model` associated to the :py:class:`clinicadl.train.Trainer`.
        maps : Maps
            The :py:class:`clinicadl.io.Maps` associated to the :py:class:`clinicadl.train.Trainer`.
        state : TrainerState
            The current :py:class:`clinicadl.train.TrainerState`.
        """

    def on_epoch_end(self, *, model: Model, maps: Maps, state: TrainerState) -> None:
        """
        Called at the end of an epoch in :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`.

        Parameters
        ----------
        model : Model
            The :py:class:`clinicadl.models.Model` associated to the :py:class:`clinicadl.train.Trainer`.
        maps : Maps
            The :py:class:`clinicadl.io.Maps` associated to the :py:class:`clinicadl.train.Trainer`.
        state : TrainerState
            The current :py:class:`clinicadl.train.TrainerState`.
        """

    def on_batch_start(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        batch: BatchType,
    ) -> None:
        """
        Called every time a new batch has been loaded in training, validation, test or prediction phases.

        .. note::
            This event may be redundant with other events: e.g., in evaluation phases, it is equivalent
            to :py:meth:`on_evaluation_start` (except if the batch is sent to another device).

        Parameters
        ----------
        model : Model
            The :py:class:`clinicadl.models.Model` associated to the :py:class:`clinicadl.train.Trainer`.
        maps : Maps
            The :py:class:`clinicadl.io.Maps` associated to the :py:class:`clinicadl.train.Trainer`.
        state : TrainerState
            The current :py:class:`clinicadl.train.TrainerState`.
        batch : BatchType
            The batch input to :py:meth:`Model.forward_step <clinicadl.models.Model.forward_step>`.
        """

    def on_batch_end(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
    ) -> None:
        """
        Called every time the processing of a batch is complete in training, validation, test or prediction phases.

        .. note::
            This event may be redundant with other events: e.g., in evaluation phases, it is equivalent
            to :py:meth:`on_evaluation_end`.

        Parameters
        ----------
        model : Model
            The :py:class:`clinicadl.models.Model` associated to the :py:class:`clinicadl.train.Trainer`.
        maps : Maps
            The :py:class:`clinicadl.io.Maps` associated to the :py:class:`clinicadl.train.Trainer`.
        state : TrainerState
            The current :py:class:`clinicadl.train.TrainerState`.
        """

    def on_forward_step_start(
        self, *, model: Model, maps: Maps, state: TrainerState, batch: BatchType
    ) -> None:
        """
        Called every time :py:meth:`Model.forward_step <clinicadl.models.Model.forward_step>` will
        be called in :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`.

        Parameters
        ----------
        model : Model
            The :py:class:`clinicadl.models.Model` associated to the :py:class:`clinicadl.train.Trainer`.
        maps : Maps
            The :py:class:`clinicadl.io.Maps` associated to the :py:class:`clinicadl.train.Trainer`.
        state : TrainerState
            The current :py:class:`clinicadl.train.TrainerState`.
        batch : BatchType
            The batch input to :py:meth:`Model.forward_step <clinicadl.models.Model.forward_step>`.
        """

    def on_backward_step_start(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        loss: LossType,
        grad_scaler: torch.amp.GradScaler,
    ) -> None:
        """
        Called every time :py:meth:`Model.backward_step <clinicadl.models.Model.backward_step>` will
        be called in :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`.

        .. note::
            This event is equivalent to ``on_forward_step_end``.

        Parameters
        ----------
        model : Model
            The :py:class:`clinicadl.models.Model` associated to the :py:class:`clinicadl.train.Trainer`.
        maps : Maps
            The :py:class:`clinicadl.io.Maps` associated to the :py:class:`clinicadl.train.Trainer`.
        state : TrainerState
            The current :py:class:`clinicadl.train.TrainerState`.
        loss : LossType
            The loss output by :py:meth:`Model.forward_step <clinicadl.models.Model.forward_step>` and
            input by :py:meth:`Model.forward_step <clinicadl.models.Model.forward_step>`.
        grad_scaler : torch.amp.GradScaler
            The :py:class:`torch.amp.GradScaler` used to scale gradients.
        """

    def on_backward_step_end(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
    ) -> None:
        """
        Called every time :py:meth:`Model.backward_step <clinicadl.models.Model.backward_step>` has just
        been called in :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`.

        Parameters
        ----------
        model : Model
            The :py:class:`clinicadl.models.Model` associated to the :py:class:`clinicadl.train.Trainer`.
        maps : Maps
            The :py:class:`clinicadl.io.Maps` associated to the :py:class:`clinicadl.train.Trainer`.
        state : TrainerState
            The current :py:class:`clinicadl.train.TrainerState`.
        """

    def on_optimization_step_start(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        optimizers: dict[str, torch.optim.Optimizer],
        grad_scaler: torch.amp.GradScaler,
    ) -> None:
        """
        Called every time :py:meth:`Model.backward_step <clinicadl.models.Model.optimization_step>` will
        be called in :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`.

        Parameters
        ----------
        model : Model
            The :py:class:`clinicadl.models.Model` associated to the :py:class:`clinicadl.train.Trainer`.
        maps : Maps
            The :py:class:`clinicadl.io.Maps` associated to the :py:class:`clinicadl.train.Trainer`.
        state : TrainerState
            The current :py:class:`clinicadl.train.TrainerState`.
        optimizers : dict[str, torch.optim.Optimizer]
            The current :py:class`Optimizer <torch.optim.Optimizer>`, returned as they are returned
            by :py:meth:`Model.backward_step <clinicadl.models.Model.build_optimizers>`
        grad_scaler : torch.amp.GradScaler
            The :py:class:`torch.amp.GradScaler` used to scale gradients.
        """

    def on_optimization_step_end(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        optimizers: dict[str, torch.optim.Optimizer],
        grad_scaler: torch.amp.GradScaler,
    ) -> None:
        """
        Called every time :py:meth:`Model.backward_step <clinicadl.models.Model.optimization_step>` has just
        been called in :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`.

        Parameters
        ----------
        model : Model
            The :py:class:`clinicadl.models.Model` associated to the :py:class:`clinicadl.train.Trainer`.
        maps : Maps
            The :py:class:`clinicadl.io.Maps` associated to the :py:class:`clinicadl.train.Trainer`.
        state : TrainerState
            The current :py:class:`clinicadl.train.TrainerState`.
        optimizers : dict[str, torch.optim.Optimizer]
            The current :py:class`Optimizer <torch.optim.Optimizer>`, returned as they are returned
            by :py:meth:`Model.backward_step <clinicadl.models.Model.build_optimizers>`
        grad_scaler : torch.amp.GradScaler
            The :py:class:`torch.amp.GradScaler` used to scale gradients.
        """

    # Evaluate

    def on_validation_start(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        split: Split,
        metrics: MetricsHandler,
        computational: ComputationalConfig,
        model_checkpoint: Optional[str] = None,
    ) -> None:
        """
        Called once at the beginning of :py:meth:`Trainer.validate <clinicadl.train.Trainer.validate>`
        or at the beginning of every validation loop in :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`

        Parameters
        ----------
        model : Model
            The :py:class:`clinicadl.models.Model` associated to the :py:class:`clinicadl.train.Trainer`.
        maps : Maps
            The :py:class:`clinicadl.io.Maps` associated to the :py:class:`clinicadl.train.Trainer`.
        state : TrainerState
            The current :py:class:`clinicadl.train.TrainerState`.
        split : Split
            The :py:class:`clinicadl.split.Split` on which validation is performed.
        metrics : MetricsHandler
            The :py:class:`~clinicadl.metrics.MetricsHandler` containing the validation metrics.
        computational : ComputationalConfig
            The :py:class:`clinicadl.train.ComputationalConfig` defining the computational specifications
            of the validation phase.
        model_checkpoint : Optional[str], default=None
            The model checkpoint currently being validated. In :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`,
            it will be ``None``.
        """

    def on_validation_end(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        metrics: MetricsHandler,
    ) -> None:
        """
        Called once at the end of :py:meth:`Trainer.validate <clinicadl.train.Trainer.validate>`
        or at the end of every validation loop in :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`

        Parameters
        ----------
        model : Model
            The :py:class:`clinicadl.models.Model` associated to the :py:class:`clinicadl.train.Trainer`.
        maps : Maps
            The :py:class:`clinicadl.io.Maps` associated to the :py:class:`clinicadl.train.Trainer`.
        state : TrainerState
            The current :py:class:`clinicadl.train.TrainerState`.
        metrics : MetricsHandler
            The :py:class:`~clinicadl.metrics.MetricsHandler` containing the validation metrics.
        """

    def on_evaluation_step_start(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        batch: BatchType,
    ) -> None:
        """
        Called every time :py:meth:`Model.evaluation_step <clinicadl.models.Model.evaluation_step>` will
        be called in :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`, :py:meth:`Trainer.validate <clinicadl.train.Trainer.validate>`,
        or :py:meth:`Trainer.test <clinicadl.train.Trainer.test>`.

        Parameters
        ----------
        model : Model
            The :py:class:`clinicadl.models.Model` associated to the :py:class:`clinicadl.train.Trainer`.
        maps : Maps
            The :py:class:`clinicadl.io.Maps` associated to the :py:class:`clinicadl.train.Trainer`.
        state : TrainerState
            The current :py:class:`clinicadl.train.TrainerState`.
        batch : BatchType
            The batch input to :py:meth:`Model.evaluation_step <clinicadl.models.Model.evaluation_step>`.
        """

    def on_evaluation_step_end(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        output: Batch,
        detailed_metrics_df: pd.DataFrame,
    ) -> None:
        """
        Called every time :py:meth:`Model.evaluation_step <clinicadl.models.Model.evaluation_step>` has just
        been called in :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`, :py:meth:`Trainer.validate <clinicadl.train.Trainer.validate>`,
        or :py:meth:`Trainer.test <clinicadl.train.Trainer.test>`.

        Parameters
        ----------
        model : Model
            The :py:class:`clinicadl.models.Model` associated to the :py:class:`clinicadl.train.Trainer`.
        maps : Maps
            The :py:class:`clinicadl.io.Maps` associated to the :py:class:`clinicadl.train.Trainer`.
        state : TrainerState
            The current :py:class:`clinicadl.train.TrainerState`.
        output : Batch
            The :py:class:`clinicadl.data.dataloader.Batch` output by :py:meth:`Model.evaluation_step <clinicadl.models.Model.evaluation_step>`.
        detailed_metrics_df : pd.DataFrame
            The evaluation metrics on the batch.
        """

    # Test

    def on_test_start(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        dataloader: DataLoader,
        metrics: MetricsHandler,
        model_checkpoint: str,
        group_name: str,
        computational: ComputationalConfig,
    ) -> None:
        """
        Called once at the beginning of :py:meth:`Trainer.test <clinicadl.train.Trainer.test>`.

        Parameters
        ----------
        model : Model
            The :py:class:`clinicadl.models.Model` associated to the :py:class:`clinicadl.train.Trainer`.
        maps : Maps
            The :py:class:`clinicadl.io.Maps` associated to the :py:class:`clinicadl.train.Trainer`.
        state : TrainerState
            The current :py:class:`clinicadl.train.TrainerState`.
        dataloader : DataLoader
            The dataloader on which the test is performed.
        metrics : MetricsHandler
            The :py:class:`~clinicadl.metrics.MetricsHandler` containing the test metrics.
        model_checkpoint : Optional[str]
            The model checkpoint currently being tested.
        group_name : str
            The name given to the test data.
        computational : ComputationalConfig
            The :py:class:`clinicadl.train.ComputationalConfig` defining the computational specifications
            of the validation phase.
        """

    def on_test_end(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        metrics: MetricsHandler,
    ) -> None:
        """
        Called once at the end of :py:meth:`Trainer.test <clinicadl.train.Trainer.test>`.

        Parameters
        ----------
        model : Model
            The :py:class:`clinicadl.models.Model` associated to the :py:class:`clinicadl.train.Trainer`.
        maps : Maps
            The :py:class:`clinicadl.io.Maps` associated to the :py:class:`clinicadl.train.Trainer`.
        state : TrainerState
            The current :py:class:`clinicadl.train.TrainerState`.
        metrics : MetricsHandler
            The :py:class:`~clinicadl.metrics.MetricsHandler` containing the test metrics.
        """

    # Predict

    def on_predict_start(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        dataloader: DataLoader,
        model_checkpoint: str,
        group_name: str,
        computational: ComputationalConfig,
    ) -> None:
        """
        Called once at the beginning of :py:meth:`Trainer.predict <clinicadl.train.Trainer.predict>`.

        Parameters
        ----------
        model : Model
            The :py:class:`clinicadl.models.Model` associated to the :py:class:`clinicadl.train.Trainer`.
        maps : Maps
            The :py:class:`clinicadl.io.Maps` associated to the :py:class:`clinicadl.train.Trainer`.
        state : TrainerState
            The current :py:class:`clinicadl.train.TrainerState`.
        dataloader : DataLoader
            The dataloader on which the prediction is performed.
        model_checkpoint : Optional[str]
            The model checkpoint currently being used.
        group_name : str
            The name given to the data.
        computational : ComputationalConfig
            The :py:class:`clinicadl.train.ComputationalConfig` defining the computational specifications
            of the prediction phase.
        """

    def on_predict_end(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
    ) -> None:
        """
        Called once at the end of :py:meth:`Trainer.predict <clinicadl.train.Trainer.predict>`.

        Parameters
        ----------
        model : Model
            The :py:class:`clinicadl.models.Model` associated to the :py:class:`clinicadl.train.Trainer`.
        maps : Maps
            The :py:class:`clinicadl.io.Maps` associated to the :py:class:`clinicadl.train.Trainer`.
        state : TrainerState
            The current :py:class:`clinicadl.train.TrainerState`.
        """

    @abstractmethod
    def state_dict(self) -> Mapping[str, Any]:
        """
        To get a checkpoint of the current state of the callback.

        Returns
        -------
        Mapping[str, Any]
            The current state in a ``dict``.
        """

    @abstractmethod
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """
        Sets to callbacks to a given state.

        Parameters
        ----------
        state_dict : Mapping[str, Any]
            The desired state of the ``Callback`, as returned by :py:meth:`state_dict`.
        """
