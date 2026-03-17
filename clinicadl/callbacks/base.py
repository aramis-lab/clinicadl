from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping

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


class Event(str, Enum):
    """Event that can trigger an action from a :py:class:`clinicadl.callbacks.Callback`."""

    EXCEPTION = "on_exception"
    INIT = "on_trainer_init"

    # Training
    TRAIN_START = "on_train_start"
    RESUME = "on_resume"
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
    VALIDATE_START = "on_validate_start"
    VALIDATE_END = "on_validate_end"
    EVAL_START = "on_evaluation_step_start"
    METRIC_START = "on_metrics_computation_start"
    METRIC_END = "on_metrics_computation_end"

    # Test
    TEST_START = "on_test_start"
    TEST_END = "on_test_end"

    # Predict
    PREDICT_START = "on_predict_start"
    PREDICTION_START = "on_prediction_step_start"
    PREDICTION_END = "on_prediction_step_end"
    PREDICT_END = "on_predict_end"


class Callback:
    """
    To define arbitrary actions to perform at certain points of the training, evaluation and prediction workflows.

    Each method of this class starting by ``on_...`` is associated to an event of
    the training, evaluation or prediction phase of the :py:class:`~clinicadl.train.Trainer`.
    By overriding these methods, the user can define action to perform when the event happens.

    .. important::
        Callbacks should capture **NON-ESSENTIAL** logic such as saving checkpoints or logging.
        The essential logic should be defined in a :py:class:`clinicadl.models.Model`.

    To define your own callback, you can override any of the method associated to an event.
    You can also override :py:meth:`state_dict` and :py:meth:`load_state_dict`, to be able
    the recover the state of your callback when resuming an interrupted training.

    .. dropdown:: List of events
        :icon: list-unordered
        :color: info

        - :py:meth:`on_trainer_init`: called when the trainer is instantiated.
        - :py:meth:`on_exception`: called whenever an uncaught exception is raised.

        .. tab-set::

            .. tab-item:: :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`

                - :py:meth:`on_backward_step_end`
                - :py:meth:`on_backward_step_start`
                - :py:meth:`on_batch_end`
                - :py:meth:`on_batch_start`
                - :py:meth:`on_epoch_end`
                - :py:meth:`on_epoch_start`
                - :py:meth:`on_evaluation_step_start`
                - :py:meth:`on_forward_step_start`
                - :py:meth:`on_metrics_computation_end`
                - :py:meth:`on_metrics_computation_start`
                - :py:meth:`on_optimization_step_end`
                - :py:meth:`on_optimization_step_start`
                - :py:meth:`on_resume`
                - :py:meth:`on_train_end`
                - :py:meth:`on_train_start`
                - :py:meth:`on_validate_end`
                - :py:meth:`on_validation_start`

                .. code-block::
                    :caption: pseudocode
                    :emphasize-lines: 2, 4, 8, 11, 15, 17, 19, 22, 24, 27, 31, 34, 38, 40, 42, 44, 47, 50, 53

                    IF resume THEN
                        CALL on_resume
                    ELSE
                        CALL on_train_start
                    END IF

                    FOR EACH epoch IN training_epochs DO
                        CALL on_epoch_start

                        FOR EACH batch in training_dataloader DO
                            CALL on_batch_start

                            MOVE batch TO appropriate_device

                            CALL on_forward_step_start
                            DO forward_step
                            CALL on_backward_step_start
                            DO backward_step
                            CALL on_backward_step_end

                            IF batch_idx MOD optimization_interval == 0 THEN
                                CALL on_optimization_step_start
                                DO optimization_step
                                CALL on_optimization_step_end
                            END IF

                            CALL on_batch_end
                        END FOR

                        IF epoch_idx MOD evaluation_interval == 0 THEN
                            CALL on_validation_start

                            FOR EACH batch in validation_dataloader DO
                                CALL on_batch_start

                                MOVE batch TO appropriate_device

                                CALL on_evaluation_step_start
                                DO evaluation_step
                                CALL on_metrics_computation_start
                                DO metrics_computation
                                CALL on_metrics_computation_end

                            CALL on_batch_end
                        END FOR

                        CALL on_validation_end
                        END IF

                        CALL on_epoch_end
                    END FOR

                    CALL on_train_end

            .. tab-item:: :py:meth:`Trainer.validate <clinicadl.train.Trainer.validate>`

                - :py:meth:`on_batch_end`
                - :py:meth:`on_batch_start`
                - :py:meth:`on_evaluation_step_start`
                - :py:meth:`on_metrics_computation_end`
                - :py:meth:`on_metrics_computation_start`
                - :py:meth:`on_validate_end`
                - :py:meth:`on_validate_start`

                .. code-block::
                    :caption: pseudocode
                    :emphasize-lines: 1, 4, 8, 10, 12, 14, 17

                    CALL on_validate_start

                    FOR EACH batch in validation_dataloader DO
                        CALL on_batch_start

                        MOVE batch TO appropriate_device

                        CALL on_evaluation_step_start
                        DO evaluation_step
                        CALL on_metrics_computation_start
                        DO metrics_computation
                        CALL on_metrics_computation_end

                        CALL on_batch_end
                    END FOR

                    CALL on_validate_end

            .. tab-item:: :py:meth:`Trainer.test <clinicadl.train.Trainer.test>`

                - :py:meth:`on_batch_end`
                - :py:meth:`on_batch_start`
                - :py:meth:`on_evaluation_step_start`
                - :py:meth:`on_metrics_computation_end`
                - :py:meth:`on_metrics_computation_start`
                - :py:meth:`on_test_end`
                - :py:meth:`on_test_start`

                .. code-block::
                    :caption: pseudocode
                    :emphasize-lines: 1, 4, 8, 10, 12, 14, 17

                    CALL on_test_start

                    FOR EACH batch in validation_dataloader DO
                        CALL on_batch_start

                        MOVE batch TO appropriate_device

                        CALL on_evaluation_step_start
                        DO evaluation_step
                        CALL on_metrics_computation_start
                        DO metrics_computation
                        CALL on_metrics_computation_end

                        CALL on_batch_end
                    END FOR

                    CALL on_test_end

            .. tab-item:: :py:meth:`Trainer.predict <clinicadl.train.Trainer.predict>`

                - :py:meth:`on_batch_end`
                - :py:meth:`on_batch_start`
                - :py:meth:`on_evaluation_step_start`
                - :py:meth:`on_metrics_computation_end`
                - :py:meth:`on_metrics_computation_start`
                - :py:meth:`on_predict_end`
                - :py:meth:`on_predict_start`

                .. code-block::
                    :caption: pseudocode
                    :emphasize-lines: 1, 4, 8, 10, 12, 15

                    CALL on_predict_start

                    FOR EACH batch in validation_dataloader DO
                        CALL on_batch_start

                        MOVE batch TO appropriate_device

                        CALL on_prediction_step_start
                        DO prediction_step
                        CALL on_prediction_step_end

                        CALL on_batch_end
                    END FOR

                    CALL on_predict_end

    """

    def state_dict(self) -> Mapping[str, Any]:
        """
        To get a checkpoint of the current state of the callback.

        Returns
        -------
        Mapping[str, Any]
            The current state in a ``dict``.
        """

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """
        Sets to callbacks to a given state.

        Parameters
        ----------
        state_dict : Mapping[str, Any]
            The desired state of the ``Callback`, as returned by :py:meth:`state_dict`.
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
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
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
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
        loss : LossType
            The loss output by :py:meth:`Model.forward_step <clinicadl.models.Model.forward_step>` and
            input by :py:meth:`Model.forward_step <clinicadl.models.Model.forward_step>`.
        grad_scaler : torch.amp.GradScaler
            The :py:class:`torch.amp.GradScaler` used to scale gradients.
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
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
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
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
        batch : BatchType
            The batch input to :py:meth:`Model.forward_step <clinicadl.models.Model.forward_step>`.
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
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
        batch : BatchType
            The batch input to :py:meth:`Model.evaluation_step <clinicadl.models.Model.evaluation_step>`.
        """

    def on_epoch_end(self, *, model: Model, maps: Maps, state: TrainerState) -> None:
        """
        Called at the end of an epoch in :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`.

        Parameters
        ----------
        model : Model
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
        """

    def on_epoch_start(self, *, model: Model, maps: Maps, state: TrainerState) -> None:
        """
        Called at the beginning of an epoch in :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`.

        Parameters
        ----------
        model : Model
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
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
            The model associated to the :py:class:`~clinicadl.train.Trainer`.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
        exception : Exception
            The exception that has been raised.
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
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
        batch : BatchType
            The batch input to :py:meth:`Model.forward_step <clinicadl.models.Model.forward_step>`.
        """

    def on_metrics_computation_end(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        detailed_metrics_df: pd.DataFrame,
    ) -> None:
        """
        Called every time metrics have just been computed on a batch.

        Parameters
        ----------
        model : Model
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
        detailed_metrics_df : pd.DataFrame
            The evaluation metrics on the batch.
        """

    def on_metrics_computation_start(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        output: Batch,
        metrics: MetricsHandler,
    ) -> None:
        """
        Called every time :py:meth:`Model.evaluation_step <clinicadl.models.Model.evaluation_step>` has been called
        be called and metrics will now be computed.

        .. note::
            This event is equivalent to ``on_evaluation_step_end``.

        Parameters
        ----------
        model : Model
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
        output : Batch
            The :py:class:`clinicadl.data.dataloader.Batch` output by :py:meth:`Model.evaluation_step <clinicadl.models.Model.evaluation_step>`.
        metrics :``be computed`.
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
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
        optimizers : dict[str, torch.optim.Optimizer]
            The current :py:class`Optimizer <torch.optim.Optimizer>`, returned as they are returned
            by :py:meth:`Model.backward_step <clinicadl.models.Model.build_optimizers>`
        grad_scaler : torch.amp.GradScaler
            The :py:class:`torch.amp.GradScaler` used to scale gradients.
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
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
        optimizers : dict[str, torch.optim.Optimizer]
            The current :py:class`Optimizer <torch.optim.Optimizer>`, returned as they are returned
            by :py:meth:`Model.backward_step <clinicadl.models.Model.build_optimizers>`
        grad_scaler : torch.amp.GradScaler
            The :py:class:`torch.amp.GradScaler` used to scale gradients.
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
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
        """

    def on_predict_start(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        dataloader: DataLoader,
        model_checkpoint: str,
        group_name: str,
        callbacks: CallbacksHandler,
        computational: ComputationalConfig,
    ) -> None:
        """
        Called once at the beginning of :py:meth:`Trainer.predict <clinicadl.train.Trainer.predict>`.

        Parameters
        ----------
        model : Model
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
        dataloader : DataLoader
            The dataloader on which the prediction is performed.
        model_checkpoint : str
            The model checkpoint currently being used.
        group_name : str
            The name given to the data.
        callbacks : CallbacksHandler
            The callbacks passed to the ``Trainer``.
        computational : ComputationalConfig
            The :py:class:`clinicadl.train.ComputationalConfig` defining the computational specifications
            of the prediction phase.
        """

    def on_prediction_step_end(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        output: Batch,
    ) -> None:
        """
        Called every time :py:meth:`Model.prediction <clinicadl.models.Model.evaluation_step>` hast just been called.

        Parameters
        ----------
        model : Model
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
        output : Batch
            The :py:class:`clinicadl.data.dataloader.Batch` output by :py:meth:`Model.evaluation_step <clinicadl.models.Model.evaluation_step>`.
        """

    def on_prediction_step_start(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        batch: BatchType,
    ) -> None:
        """
        Called every time :py:meth:`Model.prediction <clinicadl.models.Model.evaluation_step>` will be called.

        Parameters
        ----------
        model : Model
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
        batch : BatchType
            The batch input to :py:meth:`Model.evaluation_step <clinicadl.models.Model.evaluation_step>`.
        """

    def on_resume(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        split: Split,
        optimizers: dict[str, torch.optim.Optimizer],
        grad_scaler: torch.amp.GradScaler,
        optimization: OptimizationConfig,
        metrics: MetricsHandler,
        callbacks: CallbacksHandler,
        computational: ComputationalConfig,
    ) -> None:
        """
        Called when :py:meth:`Trainer.train <clinicadl.train.Trainer.train>` is resuming a training.

        More precisely, this method will be called just before loading the checkpoints.

        Parameters
        ----------
        model : Model
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
        split : Split
            The :py:class:`clinicadl.split.Split` on which training is performed.
        optimizers : dict[str, torch.optim.Optimizer]
            The :py:class`Optimizer <torch.optim.Optimizer>` returned
            by :py:meth:`Model.backward_step <clinicadl.models.Model.build_optimizers>`.
        grad_scaler : torch.amp.GradScaler
            The :py:class:`torch.amp.GradScaler` used to scale gradients.
        optimization : OptimizationConfig
            The optimization specifications of the training phase.``validation metrics`.
        callbacks : CallbacksHandler
            The callbacks passed to the ``Trainer``.
        computational : ComputationalConfig
            The :py:class:`clinicadl.train.ComputationalConfig` defining the computational specifications
            of the training phase.
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
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.``test metrics`.
        """

    def on_test_start(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        dataloader: DataLoader,
        model_checkpoint: str,
        metrics: MetricsHandler,
        group_name: str,
        callbacks: CallbacksHandler,
        computational: ComputationalConfig,
    ) -> None:
        """
        Called once at the beginning of :py:meth:`Trainer.test <clinicadl.train.Trainer.test>`.

        Parameters
        ----------
        model : Model
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
        dataloader : DataLoader
            The dataloader on which the test is performed.
        model_checkpoint : str
            The model checkpoint currently being tested.``test metrics`.
        group_name : str
            The name given to the test data.
        callbacks : CallbacksHandler
            The callbacks passed to the ``Trainer``.
        computational : ComputationalConfig
            The :py:class:`clinicadl.train.ComputationalConfig` defining the computational specifications
            of the validation phase.
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
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
        """

    def on_train_start(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        split: Split,
        optimizers: dict[str, torch.optim.Optimizer],
        grad_scaler: torch.amp.GradScaler,
        optimization: OptimizationConfig,
        metrics: MetricsHandler,
        callbacks: CallbacksHandler,
        computational: ComputationalConfig,
    ) -> None:
        """
        Called once at the beginning of :py:meth:`Trainer.train <clinicadl.train.Trainer.train>` if ``resume=False``.

        If resuming a training, :py:meth:`on_resume` will be called instead.

        Parameters
        ----------
        model : Model
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
        split : Split
            The :py:class:`clinicadl.split.Split` on which training is performed.
        optimizers : dict[str, torch.optim.Optimizer]
            The :py:class`Optimizer <torch.optim.Optimizer>` returned
            by :py:meth:`Model.backward_step <clinicadl.models.Model.build_optimizers>`.
        grad_scaler : torch.amp.GradScaler
            The :py:class:`torch.amp.GradScaler` used to scale gradients.
        optimization : OptimizationConfig
            The optimization specifications of the training phase.``validation metrics`.
        callbacks : CallbacksHandler
            The callbacks passed to the ``Trainer``.
        computational : ComputationalConfig
            The :py:class:`clinicadl.train.ComputationalConfig` defining the computational specifications
            of the training phase.
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
        Called once when the :py:class:`~clinicadl.train.Trainer` is created or restored with
        :py:meth:`Trainer.from_maps <clinicadl.train.Trainer.from_maps>`.

        Parameters
        ----------
        model : Model
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
        metrics : MetricsHandler
            The the metrics passed to the ``Trainer``.
        optimization : OptimizationConfig
            The optimization specifications of the training phase.
        callbacks : CallbacksHandler
            The callbacks passed to the ``Trainer``.
        """

    def on_validate_end(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        metrics: MetricsHandler,
    ) -> None:
        """
        Called at the end of a model validation in :py:meth:`Trainer.validate <clinicadl.train.Trainer.validate>`.

        Not to be confused with :py:meth:`on_validation_end`.

        Parameters
        ----------
        model : Model
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.``validation metrics`.
        """

    def on_validate_start(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        dataloader: DataLoader,
        model_checkpoint: str,
        metrics: MetricsHandler,
        callbacks: CallbacksHandler,
        computational: ComputationalConfig,
    ) -> None:
        """
        Called when a model is validated in :py:meth:`Trainer.validate <clinicadl.train.Trainer.validate>`.

        Not to be confused with :py:meth:`on_validation_start`.

        .. important::
            If ``model_checkpoint=None`` was passed to :py:meth:`Trainer.validate <clinicadl.train.Trainer.validate>`,
            all the models saved during training will be validated. Therefore, ``on_validate_start`` will be
            called for each model.

        Parameters
        ----------
        model : Model
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
        dataloader : DataLoader
            The dataloader on which validation is performed.
        model_checkpoint : str
            The model checkpoint currently being validated.``validation metrics`.
        callbacks : CallbacksHandler
            The callbacks passed to the ``Trainer``.
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
        metrics: MetricsHandler,
    ) -> None:
        """
        Called at the end of every validation loop in :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`.

        Not to be confused with :py:meth:`on_validate_end`.

        Parameters
        ----------
        model : Model
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.``validation metrics`.
        """

    def on_validation_start(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        dataloader: DataLoader,
        metrics: MetricsHandler,
    ) -> None:
        """
        Called at the beginning of every validation loop in :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`.

        Not to be confused with :py:meth:`on_validate_start`.

        Parameters
        ----------
        model : Model
            The model associated to the ``Trainer``.
        maps : Maps
            The :term:`MAPS` associated to the ``Trainer``.
        state : TrainerState
            The current state of the ``Trainer``.
        dataloader : DataLoader
            The dataloader on which validation is performed.``validation metrics`.
        """
