from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

import pandas as pd
import torch
from monai.metrics.metric import CumulativeIterationMetric as MonaiMetric
from torch.amp import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

from clinicadl.callbacks.handler import Callback, _CallbacksHandler
from clinicadl.data.dataloader import Batch, BatchType
from clinicadl.data.datasets import CapsDataset
from clinicadl.dictionary.utils import SEP
from clinicadl.io.maps.maps import Maps
from clinicadl.losses.config import LossConfig
from clinicadl.losses.types import Loss
from clinicadl.metrics.config import LossMetricConfig, MetricConfig
from clinicadl.metrics.handler import LossMetricConfig, MetricsHandler
from clinicadl.metrics.types import MetricOrConfig
from clinicadl.models import ClinicaDLModel
from clinicadl.optim.config import OptimizationConfig
from clinicadl.predictor.predictor import Predictor
from clinicadl.split.split import Split
from clinicadl.train.computational import ComputationalConfig
from clinicadl.train.trainer_state import TrainerState
from clinicadl.transforms.handlers import Postprocessing, Transforms
from clinicadl.utils.exceptions import ClinicaDLConfigurationError
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
    It leverages ClinicaDL's components like :py:class:`~clinicadl.models.clinicadl_model.ClinicaDLModel`
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
    model : :py:class:`~clinicadl.models.ClinicaDLModel`
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

    Examples
    --------
    .. code-block:: python

        preprocessing_t1 = T1Linear()
        transforms_image = Transforms()

        dataset_t1_image = CapsDataset(
            caps_directory=caps_directory,
            data=sub_ses_t1,
            preprocessing=preprocessing_t1,
            transforms=transforms_image,
            label="diagnosis",
        )
        dataset_t1_image.to_tensors(json_name="test_bis_im.json", n_proc=2)
        splitter = KFold(fold_dir)

        optim_config = OptimizationConfig(epochs=2)
        comp_config = ComputationalConfig(gpu=False)
        dataloader_config = DataLoaderConfig(batch_size=3)

        model = ClinicaDLModel(
            network=get_network_config(
                ImplementedNetwork.RESNET, num_outputs=1, spatial_dims=3, in_channels=1
            ),
            loss = MSELossConfig(),
            optimizer=AdamConfig(),
        )

        metrics = {
            "mae": MAEMetric(),
            "mse": MSEMetricConfig(),
            "matrix": ConfusionMatrixMetricConfig(metric_name=["tpr", "fpr"]
            }

        callbacks = [
            EarlyStopping(metrics=["mae", "loss"]),
            ModelSelection(metrics=["mae"]),
            EarlyStopping(metrics=["mse"]),
            CodeCarbon(),
        ]

        trainer = Trainer(
            maps_path,
            model=model,
            comp_config=comp_config,
            optim_config=optim_config,
            callbacks=callbacks,
            metrics=metrics,
            _overwrite=True,
        )

        for split in splitter.get_splits(dataset=dataset_t1_image):
            split.build_train_loader(dataloader_config)
            split.build_val_loader(dataloader_config)

            trainer.train(split)

    Notes
    -----
    .. note:
        - Training utilizes automatic mixed precision (AMP) if enabled in :py:class:`~clinicadl.utils.computational.config.ComputationalConfig`.
        - The callback system provides hooks to extend training behavior without altering core code.
        - The :py:class:`~clinicadl.train.trainer.Trainer`: expects datasets and models compatible with ClinicaDL interfaces.
        - Metrics can be dynamically updated during evaluation and training.

    """

    def __init__(
        self,
        maps_path: PathType,
        model: ClinicaDLModel,
        callbacks: Optional[list[Callback]] = None,
        metrics: dict[str, MetricOrConfig] = {
            "loss": LossMetricConfig(loss_name="loss")
        },
        optim_config: OptimizationConfig = OptimizationConfig(),
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
            model = ClinicaDLModel.from_json(maps.model_json)

        self._state: TrainerState = TrainerState(num_epochs=self._optim_config.epochs)

        self._model: ClinicaDLModel = model
        self._maps: Maps = maps
        self._metrics_handler: MetricsHandler = metrics

        self._optim_config: OptimizationConfig = optim_config

        self._scaler: torch.amp.GradScaler = comp_config.get_scaler()

        seed_everything(seed=seed, deterministic=False, compensation="memory")

    @property
    def model(self):
        return self._model

    @property
    def maps(self):
        return self._maps

    @property
    def metrics(self):
        return self._metrics_handler.df

    @property
    def detailed_metrics(self) -> pd.DataFrame:
        return self._metrics_handler.detailed_df

    @property
    def optimization_config(self) -> OptimizationConfig:
        return self._optim_config

    @property
    def state(self) -> TrainerState:
        return self._state

    @classmethod
    def from_maps(cls, maps_path: PathType):
        maps = Maps(maps_path)
        maps.load()

        model = ClinicaDLModel.from_json(maps.model_json)
        comp_config = ComputationalConfig.from_json(maps.training.computational_json)
        optim_config = OptimizationConfig.from_json(maps.training.optimization_json)
        callbacks = _CallbacksHandler.from_json(maps.training.callbacks_json)
        metrics = MetricsHandler.from_json(maps.training.metrics_json)

        # TODO : check seed ?

        return cls(
            maps_path=maps_path,
            model=model,
            callbacks=callbacks,
            metrics=metrics,  # type: ignore
            optim_config=optim_config,
            comp_config=comp_config,
            resume=True,
        )

    def reset(self):
        self.state.reset()
        self._metrics_handler.reset(reset_df=True)

    def train(
        self,
        split: Split,
        resume: bool = False,
        computational: ComputationalConfig = ComputationalConfig(),
    ) -> None:
        """
        Run the training loop over the given data split.

        Parameters
        ----------
        split : Split
            The data split containing training and validation DataLoaders.
        """
        # seed?
        self._check_split(split)
        self.model.train()  # reset model
        self.model.to(
            computational.device,
            non_blocking=computational.non_blocking,
            memory_format=torch.channels_last_3d,
        )
        split.train_loader.eval()
        self.reset()
        self._write_training_infos(split=split)

        self._call_event("on_train_begin", split=split)

        while not self.state.should_stop:
            self.state.current_epoch += 1

            self._call_event("on_epoch_begin")

            split.train_loader.set_epoch(self.state.current_epoch)

            for batch_idx, batch in enumerate(split.train_loader, start=1):
                self.state.current_train_batch = batch_idx

                self._send_to_device(batch)

                self._call_event("on_forward_step_begin", batch=batch)

                with autocast(
                    device_type=computational.device.type,
                    enabled=computational.amp,
                ):
                    loss = self.model.forward_step(batch=batch)

                self._call_event("on_forward_step_end", batch=batch, loss=loss)

                if batch_idx % self.optimization_config.accumulation_steps == 0:
                    self._call_event("on_optimization_step_begin", loss=loss)

                    self.model.optimization_step(loss, self._scaler)
                    self.state.optim_step += 1

                    self._scaler.update()
                    for optimizer in self.model.get_optimizers().values():
                        optimizer.zero_grad(set_to_none=True)

                    self._call_event(
                        "on_optimization_step_end",
                        optimizers=self.model.get_optimizers(),
                        grad_scaler=self._scaler,
                    )

            if (
                self.state.current_epoch % self.optimization_config.evaluation_steps
                == 0
            ):
                self.evaluate(split.val_loader)

            self._call_event("on_epoch_end")

            if self.state.current_epoch == self.state.num_epochs:
                self.state.should_stop = True

        self._write_end_training_infos(split=split)

        self._call_event("on_train_end")

        self._clear_tmp()

    def evaluate(
        self,
        split: Split,
        computational: ComputationalConfig = ComputationalConfig(),
        metrics: Optional[dict[str, MetricOrConfig]] = None,
    ) -> None:
        """
        Evaluate the model on a validation or test dataset.
        """
        self._check_split(split, only_val=True)
        self.model.eval()
        self.model.to(computational.device)
        split.val_loader.dataset.eval()

        if metrics:
            self._metrics_handler.add_metrics(
                **metrics
            )  # no need to recompute the other ones

        self._reset_validation()

        self._call_event("on_evaluate_begin", split=split)

        with torch.no_grad():
            for batch_idx, batch in enumerate(split.val_loader):
                self.state.current_val_batch = batch_idx

                self._send_to_device(batch)

                self._call_event("on_evaluation_step_begin", batch=batch)

                output_batch = self.model.evaluation_step(
                    batch
                )  # amp during evaluation

                metrics = self._metrics_handler(
                    output_batch, epoch=self.state.current_epoch
                )

                self._call_event(
                    "on_evaluation_step_end",
                    batch=batch,
                    output=output_batch,
                    metrics=metrics,
                )

        self._metrics_handler.aggregate(epoch=self.state.current_epoch)

        self._metrics_handler.save(
            path=self.maps.training.splits[split.index].validation_metrics.aggregated,
            details_path=self.maps.training.splits[
                split.index
            ].validation_metrics.aggregated,
        )

        self._call_event(
            "on_evaluate_end",
            metrics=self._metrics_handler.df,
            detailed_metrics=self._metrics_handler.detailed_df,
        )

    def predict(
        self,
        dataloder: DataLoader,
        model: str,
        group_name: str,
        computational: ComputationalConfig = ComputationalConfig(),
        metrics: Optional[dict[str, MetricOrConfig]] = None,
    ) -> None:
        """
        Evaluate the model on a validation or test dataset.
        """
        self.model.eval()
        split, criterion = self._read_model_str(model)
        # load the right model here
        self.model.to(computational.device)
        dataloder.dataset.eval()
        self._check_group(group_name)

        metrics_handler = MetricsHandler()  # no need to recompute the other ones

        self._reset_prediction()

        self._call_event("on_predict_begin")

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloder):
                self.state.current_val_batch = batch_idx

                self._send_to_device(batch)

                self._call_event("on_evaluation_step_begin", batch=batch)

                output_batch = self.model.evaluation_step(batch)

                metrics = self._metrics_handler(
                    output_batch, epoch=self.state.current_epoch
                )

                self._call_event(
                    "on_evaluation_step_end",
                    batch=batch,
                    output=output_batch,
                    metrics=metrics,
                )

        self._metrics_handler.aggregate(epoch=self.state.current_epoch)

        self._metrics_handler.save(
            path=self.maps.predictions.groups[group_name]
            .splits[split]
            .best_models[criterion]
            .metrics.aggregated,
            details_path=self.maps.predictions.groups[group_name]
            .splits[split]
            .best_models[criterion]
            .metrics.details,
        )

        self._call_event(
            "on_predict_end",
            metrics=self._metrics_handler.df,
            detailed_metrics=self._metrics_handler.detailed_df,
        )

    def _reset_validation(self) -> None:
        self.state.reset_validation()
        self._metrics_handler.reset(reset_df=False)

    def _reset_prediction(self) -> None:
        self.state.reset_prediction()
        self._metrics_handler.reset(reset_df=True)

    def _send_to_device(self, data: BatchType) -> None:
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

    def _read_model_str(self, model: str) -> tuple[int, str]:
        raise NotImplementedError()

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
