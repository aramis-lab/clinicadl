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
    from clinicadl.models import Model
    from clinicadl.optim.config import OptimizationConfig
    from clinicadl.split import Split
    from clinicadl.train import TrainerState
    from clinicadl.train.computational import ComputationalConfig


class Events(str, Enum):
    """Events that can trigger an action from a :py:class:`clinicadl.callbacks.Callback`."""

    # Training
    TRAIN_BEGIN = "on_train_begin"
    TRAIN_END = "on_train_end"
    EPOCH_BEGIN = "on_epoch_begin"
    EPOCH_END = "on_epoch_end"
    FORWARD_BEGIN = "on_forward_step_begin"
    BACKWARD_BEGIN = "on_backward_step_begin"
    BACKWARD_END = "on_backward_step_end"
    OPTIM_STEP_BEGIN = "on_optimization_step_begin"
    OPTIM_STEP_END = "on_optimization_step_end"

    # Validation
    VAL_BEGIN = "on_validation_begin"
    VAL_END = "on_validation_end"
    EVAL_BEGIN = "on_evaluation_step_begin"
    EVAL_END = "on_evaluation_step_end"

    # Test
    TEST_BEGIN = "on_test_begin"
    TEST_END = "on_test_end"

    # Predict
    PREDICT_BEGIN = "on_test_begin"
    PREDICT_END = "on_test_end"
    PREDICTION_BEGIN = "on_prediction_step_begin"
    PREDICTION_END = "on_prediction_step_end"


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

    # Train

    def on_train_begin(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        split: Split,
        optimizers: dict[str, torch.optim.Optimizer],
        computational: ComputationalConfig,
        optimization: OptimizationConfig,
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
        computational : ComputationalConfig
            The :py:class:`clinicadl.train.ComputationalConfig` defining the computational specifications
            of the training phase.
        optimization : OptimizationConfig
            The :py:class:`clinicadl.optim.OptimizationConfig` defining the optimization specifications
            of the training phase.
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

    def on_epoch_begin(self, *, model: Model, maps: Maps, state: TrainerState) -> None:
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

    def on_forward_step_begin(
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

    def on_backward_step_begin(
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

    def on_optimization_step_begin(
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

    def on_validation_begin(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        split: Split,
        model_checkpoint: Optional[str],
        computational: ComputationalConfig,
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
        model_checkpoint : Optional[str]
            The model checkpoint currently being validated. In :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`,
            it will be ``None``.
        computational : ComputationalConfig
            The :py:class:`clinicadl.train.ComputationalConfig` defining the computational specifications
            of the validation phase.
        """

    def on_validation_end(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        metrics: pd.DataFrame,
        detailed_metrics: pd.DataFrame,
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
        metrics : pd.DataFrame
            The aggregated validation metrics.
        detailed_metrics : pd.DataFrame
            The detailed validation metrics (i.e. the metrics for each image).
        """

    def on_evaluation_step_begin(
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
        metrics: pd.DataFrame,
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
        metrics : pd.DataFrame
            The evaluation metrics on the batch.
        """

    # Test

    def on_test_begin(
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
        metrics: pd.DataFrame,
        detailed_metrics: pd.DataFrame,
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
        metrics : pd.DataFrame
            The aggregated test metrics.
        detailed_metrics : pd.DataFrame
            The detailed test metrics (i.e. the metrics for each image).
        """

    # Predict

    def on_predict_begin(
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

    def on_prediction_end(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
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
