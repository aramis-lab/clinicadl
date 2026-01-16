from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from monai.metrics.metric import CumulativeIterationMetric as MonaiMetric
from torch.amp import GradScaler
from torch.amp.autocast_mode import autocast

from clinicadl.callbacks.handler import Callback, _CallbacksHandler
from clinicadl.data.dataloader import Batch, BatchType, DataLoader
from clinicadl.data.datasets import CapsDataset, Dataset
from clinicadl.io.maps.maps import Maps
from clinicadl.io.maps.training.splits.models import ModelDir
from clinicadl.losses.config import LossConfig
from clinicadl.losses.types import Loss
from clinicadl.metrics.config import LossMetricConfig, MetricConfig
from clinicadl.metrics.handler import LossMetricConfig, MetricsHandler
from clinicadl.metrics.types import MetricOrConfig
from clinicadl.modelss import Model
from clinicadl.optim.config import OptimizationConfig
from clinicadl.predictor.predictor import Predictor
from clinicadl.split.split import Split
from clinicadl.train.computational import ComputationalConfig
from clinicadl.train.trainer_state import TrainerStage, TrainerState
from clinicadl.transforms.handlers import Postprocessing, Transforms
from clinicadl.utils.dictionary.utils import SEP
from clinicadl.utils.dictionary.words import PARTICIPANT_ID, SESSION_ID
from clinicadl.utils.exceptions import ClinicaDLConfigurationError, DataLeakageError
from clinicadl.utils.json import write_json
from clinicadl.utils.names import camel_to_snake
from clinicadl.utils.seed import seed_everything
from clinicadl.utils.typing import PathType


class Trainer:
    """
    Trainer class to manage the full lifecycle of model **training**, **evaluation**, and **prediction**
    within the ClinicaDL framework.

    This class encapsulates the training loop, evaluation, and prediction processes while
    integrating callback management, metric tracking, and mixed precision training support.
    It leverages ClinicaDL's components like :py:class:`~clinicadl.modelss.clinicadl_model.Model`
    and :py:class:`~clinicadl.io.maps.maps.Maps`,
    promoting modularity and extensibility primarily through callbacks.

    The Trainer follows a callback-driven design pattern: it invokes callbacks at key stages
    (e.g., training start/end, epoch start/end, batch start/end, backward passes) to enable
    flexible monitoring, logging, early stopping, and other behaviors without modifying
    the core training code.

    .. note:
        This class should generally not be subclassed; custom behavior should be implemented via callbacks.


    Parameters
    ----------
    maps_path : PathType
        Directory path where training outputs, maps, and metrics will be saved.
    model : :py:class:`~clinicadl.modelss.Model`
        The deep learning model to train and evaluate.
    callbacks : list[:py:class:`~clinicadl.callbacks.base.Callback`], optional
        List of callback instances to execute during training and evaluation.
        Defaults to None (no callbacks).
    metrics : dict[str, MetricType], optional
        Dictionary of metric names and metric instances for monitoring model performance.
        Defaults to None.
    optim_config : :py:class:`~clinicadl.optim.config.OptimizationConfig`, optional
        Configuration object specifying optimizer settings and training schedule.
        Defaults to `OptimizationConfig()`.
    comp_config : :py:class:`~clinicadl.utils.computational.config.ComputationalConfig`, optional
        Configuration for computation environment (e.g., device type, mixed precision).
        Defaults to `ComputationalConfig()`.
    _overwrite : bool, optional
        Whether to overwrite existing output files in `maps_path`.
        Defaults to False.
    seed : int, optional
        Random seed for reproducibility.
        Defaults to 123.

    """

    def __init__(
        self,
        maps_path: PathType,
        model: Model,
        metrics: dict[str, MetricOrConfig] = {
            "loss": LossMetricConfig(loss_name="loss")
        },
        optimization: OptimizationConfig = OptimizationConfig(),
        callbacks: Optional[list[Callback]] = None,
        _overwrite: bool = False,
    ) -> None:
        maps = Maps(maps_path)
        if not resume:
            train_metrics = MetricsHandler(**metrics)

            self.callbacks = _CallbacksHandler(
                metrics=train_metrics,
                callbacks=callbacks if callbacks is not None else [],
            )
            maps.create(overwrite=_overwrite)

            model.write_json(maps.model_json)
            model.write_architecture_log(maps.architecture_log)

            self.callbacks.write_json(maps.training.callbacks_json)
            train_metrics.write_json(maps.training.metrics_json)
            comp_config.write_json(maps.training.computational_json)
            optim_config.write_json(maps.training.optimization_json)

        else:
            maps.load()
            train_metrics = MetricsHandler.from_json(maps.training.metrics_json)
            self.callbacks = _CallbacksHandler.from_json(maps.training.callbacks_json)
            optim_config = OptimizationConfig.from_json(maps.training.optimization_json)
            comp_config = ComputationalConfig.from_json(
                maps.training.computational_json
            )
            model = Model.from_json(maps.model_json)

        self._state: TrainerState = TrainerState(num_epochs=self._optim_config.epochs)

        self._model: Model = model
        self._maps: Maps = maps
        self._metrics_handler: MetricsHandler = MetricsHandler(**metrics)
        self._metrics_handler.init_metrics(self._model)

        self._optim_config: OptimizationConfig = optim_config

        seed_everything(seed=seed, deterministic=False, compensation="memory")

    @property
    def model(self):
        return self._model

    @property
    def maps(self):
        return self._maps

    @property
    def metrics(self):
        return list(self._metrics_handler.metrics.keys())

    @property
    def optimization_config(self) -> OptimizationConfig:
        return self._optim_config

    @property
    def state(self) -> TrainerState:
        return self._state

    @classmethod
    def from_maps(cls, maps_path: PathType):
        maps = Maps(maps_path)
        maps.read()

        model = Model.from_json(maps.model_json)
        optim_config = OptimizationConfig.from_json(maps.training.optimization_json)
        metrics = MetricsHandler.from_json(maps.metrics_json)
        callbacks = _CallbacksHandler.from_json(maps.training.callbacks_json)

        # TODO : check seed ?

        return cls(
            maps_path=maps_path,
            model=model,
            callbacks=callbacks,
            metrics=metrics,  # type: ignore
            optim_config=optim_config,
        )

    def reset(self):
        self.state = TrainerState()
        self._metrics_handler.reset(reset_df=True)

    def add_callbacks(self, callbacks: Sequence[Callback]):
        pass

    def add_metrics(self, metrics: dict[str, MetricOrConfig]):
        self._metrics_handler.add_metrics(**metrics)

    def train(
        self,
        split: Split,
        resume: bool = False,
        computational: ComputationalConfig = ComputationalConfig(),
        reset: bool = True,
    ) -> None:
        """
        Run the training loop over the given data split.

        Parameters
        ----------
        split : Split
            The data split containing training and validation DataLoaders.
        """
        # seed?
        self.maps.read()
        self._check_split(split)
        self.model.train()  # reset model
        self.model.to(
            computational.device,
            non_blocking=computational.non_blocking,
            memory_format=torch.channels_last_3d,
        )
        split.train_loader.eval()
        self.reset()
        scaler = computational.get_scaler()
        self._write_training_infos(split=split)

        self._call_event(
            "on_train_start",
            split=split,
            computational=computational,
            optimization=self.optimization_config,
            resume=resume,
        )

        while not self.state.should_stop:
            self.state.current_epoch += 1

            self._call_event("on_epoch_start")

            split.train_loader.set_epoch(self.state.current_epoch)

            for batch_idx, batch in enumerate(split.train_loader, start=1):
                self.state.current_train_batch = batch_idx

                self._call_event("on_batch_start")

                self._send_to_device(batch)

                self._call_event("on_forward_step_start", batch=batch)

                with autocast(
                    device_type=computational.device.type,
                    enabled=computational.amp,
                ):
                    loss = self.model.forward_step(batch=batch)

                self._call_event(
                    "on_backward_step_start", loss=loss, grad_scaler=scaler
                )

                self.model.backward_step(loss, grad_scaler=scaler)

                self._call_event("on_backward_step_end")

                if batch_idx % self.optimization_config.accumulation_steps == 0:
                    self._call_event(
                        "on_optimization_step_start",
                        optimizers=self.model.get_optimizers(),
                        grad_scaler=scaler,
                    )

                    self.model.optimization_step(grad_scaler=scaler)
                    self.state.optim_step += 1

                    scaler.update()
                    for optimizer in self.model.get_optimizers().values():
                        optimizer.zero_grad(set_to_none=True)

                    self._call_event(
                        "on_optimization_step_end",
                        optimizers=self.model.get_optimizers(),
                        grad_scaler=scaler,
                    )

                self._call_event("on_batch_end")

            if (
                self.state.current_epoch - 1 % self.optimization_config.evaluation_steps
                == 0
            ):
                self._validate(split)

                self._metrics_handler.save(
                    path=self.maps.training.splits[
                        split.index
                    ].validation_metrics.aggregated,
                    details_path=self.maps.training.splits[
                        split.index
                    ].validation_metrics.aggregated,
                )

            self._call_event("on_epoch_end")

            self._save_checkpoint()

            if self.state.current_epoch == self.state.num_epochs:
                self.state.should_stop = True

        self._write_end_training_infos(split=split)

        self._call_event("on_train_end")

        self.maps.training.splits[split.index].tmp.clear()

    def validate(
        self,
        split: Split,
        model_checkpoint: str,
        metrics: Optional[Sequence[str]] = None,
        computational: ComputationalConfig = ComputationalConfig(),
    ) -> None:
        """
        Evaluate the model on a validation or test dataset.
        """
        self.maps.read()
        self._check_split(split, only_val=True)
        self._check_metrics(metrics)

        checkpoint_path = self._read_checkpoint_name(model_checkpoint)
        self._load_model_checkpoint(checkpoint_path.model)

        self.model.to(computational.device, non_blocking=computational.non_blocking)

        self._validate(
            split,
            metrics=metrics,
            computational=computational,
            model_checkpoint=model_checkpoint,
        )

        self._metrics_handler.merge(
            path=checkpoint_path.validation_metrics.aggregated,
            details_path=checkpoint_path.validation_metrics.aggregated,
        )

    def test(
        self,
        dataloader: DataLoader,
        model_checkpoint: str,
        group_name: str,
        metrics: Optional[Sequence[str]] = None,
        save_outputs: bool = False,
        computational: ComputationalConfig = ComputationalConfig(),
    ) -> None:
        """
        Evaluate the model on a validation or test dataset.
        """
        self.maps.read()
        self._check_leakage(dataloader)
        self._check_metrics(metrics)
        self._check_group(group_name)

        checkpoint_path = self._read_checkpoint_name(model_checkpoint)
        self._load_model_checkpoint(checkpoint_path.model)

        self.maps.predictions.groups[group_name].results.create_model(model)

        self._reset_test()

        self._call_event(
            "on_test_start",
            dataloader=dataloader,
            model_checkpoint=model_checkpoint,
            group_name=group_name,
            computational=computational,
        )

        self._evaluation_loop(dataloader)

        self._call_event(
            "on_test_end",
            metrics=self._metrics_handler.df,
            detailed_metrics=self._metrics_handler.detailed_df,
        )

        self._metrics_handler.save(
            path=self.maps.predictions.groups[group_name]
            .results.models[model]
            .metrics.aggregated,
            details_path=self.maps.predictions.groups[group_name]
            .results.models[model]
            .metrics.details,
        )

    def predict(
        self,
        dataloader: DataLoader,
        model_checkpoint: str,
        group_name: str,
        computational: ComputationalConfig = ComputationalConfig(),
    ) -> None:
        """
        Predict
        """
        self.maps.read()

        self._call_event(
            "on_prediction_start",
            dataloader=dataloader,
            model_checkpoint=model_checkpoint,
            group_name=group_name,
            computational=computational,
        )

        self._call_event(
            "on_prediction_end",
        )

    def _validate(
        self,
        split: Split,
        metrics: Optional[Sequence[str]],
        computational: ComputationalConfig,
        model_checkpoint: Optional[str] = None,
    ) -> None:
        self._reset_validation()

        self._call_event(
            "on_validation_start",
            split=split,
            model_checkpoint=model_checkpoint,
            computational=computational,
        )

        self._evaluation_loop(split.val_loader, metrics=metrics)

        self._call_event(
            "on_validation_end",
            metrics=self._metrics_handler.df,
            detailed_metrics=self._metrics_handler.detailed_df,
        )

    def _evaluation_loop(
        self, dataloader: DataLoader, metrics: Optional[Sequence[str]] = None
    ) -> None:
        self.model.eval()
        dataloader.dataset.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                self.state.current_val_batch = batch_idx

                self._send_to_device(batch)

                self._call_event("on_evaluation_step_start", batch=batch)

                output_batch = self.model.evaluation_step(
                    batch
                )  # amp during evaluation?

                metrics = self._metrics_handler(
                    output_batch, epoch=self.state.current_epoch, metrics=metrics
                )

                self._call_event(
                    "on_evaluation_step_end",
                    output=output_batch,
                    metrics=metrics,
                )

        self._metrics_handler.aggregate(epoch=self.state.current_epoch, metrics=metrics)

    def _reset_resume(self) -> None:
        self.state.current_train_batch = 0
        self.state.current_val_batch = 0
        self.state.stage = TrainerStage.TRAIN

    def _reset_train(self, split: Split, num_epochs: int) -> None:
        self.state.reset_training(split=split, num_epochs=num_epochs)
        self._metrics_handler.reset(reset_df=True)

    def _reset_validation(self, split: Split) -> None:
        self.state.reset_validation(split=split)
        self._metrics_handler.reset(reset_df=False)

    def _reset_validate(self, split: Split) -> None:
        self.state.reset_validation(split=split)
        self._metrics_handler.reset(reset_df=True)

    def _reset_test(self, dataloader: DataLoader) -> None:
        self.state.reset_test(dataloader=dataloader)
        self._metrics_handler.reset(reset_df=True)

    def _reset_prediction(self, dataloader: DataLoader) -> None:
        self.state.reset_test(dataloader=dataloader)

    def _check_metrics(self, metrics: Optional[Sequence[str]]) -> None:
        if metric is not None:
            for metric in metrics:
                if metric not in self._metrics_handler.metrics:
                    raise ValueError(
                        f"'{metric}' does not match any metrics. Metrics defined are: {self.metrics} "
                        "Use 'add_metrics' to define new metrics."
                    )

    def _check_leakage(self, dataloader: DataLoader) -> None:
        training_data = self.maps.open_file(self.maps.training.data.data_tsv)
        training_participants = set(training_data[PARTICIPANT_ID])
        dataset: Dataset = dataloader.dataset
        new_participants = set(zip(*dataset.get_participant_session_couples())[0])
        if len(training_participants.intersection(new_participants)) > 0:
            raise DataLeakageError(
                participants=training_participants.intersection(new_participants)
            )

    def _get_all_models(
        self, split_idx: int, final: bool, checkpoints: bool
    ) -> list[ModelDir]:
        models_path = self.maps.training.splits[split_idx].models

        all_models = list(models_path.best_models.iterdir())

        if final:
            all_models.append(models_path.final)

        if checkpoints:
            all_models.extend(models_path.checkpoints.iterdir())

    @staticmethod
    def _model_to(model: torch.nn.Module, comp_config: ComputationalConfig) -> None:
        model.to(device=comp_config.device, non_blocking=comp_config.non_blocking)
        if comp_config.channels_last:
            try:
                model.to(torch.channels_last)
            except RuntimeError:
                model.to(torch.channels_last_3d)

    @staticmethod
    def _batch_to(data: BatchType) -> None:
        """
        Send the data to the right device.
        """
        if isinstance(data, Batch):
            data.to(self.computational_config.device)
        else:
            for batch in data:
                batch.to(self.computational_config.device)

    def _call_event(self, event: str, **kwargs):
        self.callbacks.call_event(
            event, model=self.model, maps=self.maps, state=self.state, **kwargs
        )

    def _load_model_checkpoint(self, model_path: Path) -> None:
        self.model.to("cpu")  # load weights on cpu
        state_dict = self.maps.open_file(model_path)
        self.model.load_state_dict(state_dict)

    def _write_training_infos(
        self,
        split: Split,
    ) -> None:
        """
        Write training information to the maps directory.

        Parameters
        ----------
        split : Split
            The data split used for training.
        """
        self.maps._create_training_split(split=split)
        self.maps._add_lines_to_summary_log(
            f"Training dataset  : {split.train_dataset.caps_reader.input_directory}"
        )

        assert isinstance(split.train_loader.dataset, CapsDataset)
        split.train_loader.dataset.write_json(
            self.maps.training.splits[split.index].caps_dataset_json, name="train"
        )
        split.train_loader_config.write_json(
            self.maps.training.splits[split.index].dataloader_json, name="train"
        )

        assert isinstance(split.val_loader.dataset, CapsDataset)
        split.val_loader.dataset.write_json(
            self.maps.training.splits[split.index].caps_dataset_json, name="val"
        )
        split.val_loader_config.write_json(
            self.maps.training.splits[split.index].dataloader_json, name="val"
        )

    def _write_end_training_infos(
        self,
        split: Split,
    ) -> None:
        """
        Write end of training information to the maps directory.

        Parameters
        ----------
        split : Split
            The data split used for training.
        """

        self.maps._add_lines_to_summary_log(
            f"Input size        : {self.model._input_size}\n"
        )
        self.maps._add_lines_to_summary_log("=" * 15)

        self.config.write_torchsummary()  # not working i don't know why

    def _check_split(self, split: Split, only_val: bool = False) -> None:
        if not only_val:
            if split.train_loader is None:
                raise ClinicaDLConfigurationError(
                    "The split has no train_loader defined. Please run `get_dataloader()`"
                )
            self.state.num_train_batches = len(split.train_loader)
        if split.val_loader is None:
            raise ClinicaDLConfigurationError(
                "The split has no train_loader defined. Please run `get_dataloader()`"
            )
        self.state.num_val_batches = len(split.val_loader)

    def _check_group(self, group_name: str) -> None:
        self.maps.predictions.create_group(group_name)

    def _read_checkpoint_name(self, checkpoint_name: str) -> ModelDir:
        pattern = re.compile(r"^split-\d+_(?:final|metric-[a-z-]+|epoch-\d+)$")
        assert bool(re.match(pattern, checkpoint_name))

        split, checkpoint_name = checkpoint_name.split("_")
        split_idx = int(split.split("-")[-1])
        models_dir = self.maps.training.splits[split_idx].models

        if checkpoint_name == "final":
            return models_dir.final
        elif checkpoint_name.startswith("metric"):
            metric = checkpoint_name.split("-")[-1]
            return models_dir.best_models.metrics[metric]
        elif checkpoint_name.startswith("epoch"):
            epoch = int(checkpoint_name.split("-")[-1])
            return models_dir.checkpoints.epochs[epoch]

    def _save_checkpoint(self):
        if self.state.current_epoch == self._last_saved_epoch:
            return
        self._last_saved_epoch = self.state.current_epoch

        tmp_dir = self.maps.training.splits[self.state.split_idx].tmp
        tmp_dir.read()
        tmp_dir.create_epoch(self.state.current_epoch)
        epoch_dir = tmp_dir.epochs[self.state.current_epoch]

        # model
        self.model.save_checkpoint(epoch_dir.model)

        # metrics
        self._metrics_handler.save(
            path=epoch_dir.validation_metrics.aggregated,
            details_path=epoch_dir.validation_metrics.details,
        )

        # trainer state
        write_json(epoch_dir, self.state.state_dict())

        # callbacks
        for name, callback in self.callbacks.callbacks.items():
            lowered_name = camel_to_snake(name)
            callback_json = epoch_dir.callbacks / lowered_name
            callback.save_checkpoint(callback_json)

        # delete old epochs
        for epoch in tmp_dir.epochs_list:
            if epoch != self.state.current_epoch:
                tmp_dir.epochs[epoch].remove()
