from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Union

import torch
from torch.amp.autocast_mode import autocast
from typing_extensions import Self

from clinicadl.callbacks import CallbacksHandler
from clinicadl.callbacks.base import Events
from clinicadl.data.dataloader import Batch, DataLoaderConfig
from clinicadl.data.datasets.factory import get_dataset_from_json
from clinicadl.io.maps.maps import Maps
from clinicadl.metrics import MetricsHandler
from clinicadl.metrics.config import LossMetricConfig
from clinicadl.models.factory import get_model_from_json
from clinicadl.optim.config import OptimizationConfig
from clinicadl.train.computational import ComputationalConfig
from clinicadl.train.trainer_state import TrainerState
from clinicadl.utils.dictionary.words import CPU
from clinicadl.utils.exceptions import CannotReadJsonError
from clinicadl.utils.seed import seed_everything

if TYPE_CHECKING:
    from clinicadl.callbacks import Callback
    from clinicadl.data.dataloader import BatchType, DataLoader
    from clinicadl.io.maps.utils import DataDir
    from clinicadl.metrics.types import MetricOrConfig
    from clinicadl.models import Model
    from clinicadl.split.split import Split
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
        metrics: Union[dict[str, MetricOrConfig], MetricsHandler] = {
            "loss": LossMetricConfig(loss_name="loss")
        },
        optimization: OptimizationConfig = OptimizationConfig(),
        callbacks: Optional[Union[list[Callback], CallbacksHandler]] = None,
        overwrite: bool = False,
    ) -> None:
        self._maps = Maps(maps_path)
        self._maps.create(overwrite=overwrite)

        self._model = model

        if isinstance(metrics, MetricsHandler):
            self._metrics = metrics
        else:
            self._metrics = MetricsHandler(**metrics)
        self._metrics.init_metrics(self._model)

        if isinstance(callbacks, CallbacksHandler):
            self._callbacks = callbacks
        else:
            self._callbacks = CallbacksHandler(callbacks=callbacks if callbacks else [])

        self._optim_config = optimization

        self._state = TrainerState()

        self._call_event(
            Events.INIT,
            metrics=self._metrics,
            optimization=self.optimization,
            callbacks=self.callbacks,
        )

    @property
    def maps(self) -> Maps:
        return self._maps

    @property
    def model(self) -> Model:
        return self._model

    @property
    def metrics(self) -> MetricsHandler:
        return self._metrics

    @property
    def callbacks(self) -> CallbacksHandler:
        return self._callbacks

    @property
    def optimization(self) -> OptimizationConfig:
        return self._optim_config

    @property
    def state(self) -> TrainerState:
        return self._state

    @classmethod
    def from_maps(cls, maps_path: PathType, **kwargs) -> Self:
        maps = Maps(maps_path)
        maps.read()
        model = get_model_from_json(maps.model_json)
        metrics = MetricsHandler.from_json(maps.model_json)
        optimization = OptimizationConfig.from_json(maps.training.optimization_json)
        callbacks = CallbacksHandler.from_json(maps.callbacks_json)

        return cls(
            maps_path=maps_path,
            model=model,
            metrics=metrics,
            optimization=optimization,
            callbacks=callbacks,
        )

    def add_metrics(self, **metrics: MetricOrConfig) -> None:
        self._metrics.add_metrics(**metrics)
        self._metrics.to_json(self._maps.metrics_json, overwrite=True)

    def add_callbacks(self, callbacks: Sequence[Callback]) -> None:
        self._callbacks.add_callbacks(callbacks)
        self._metrics.to_json(self._maps.callbacks_json, overwrite=True)

    def train(
        self,
        split: Split,
        computational: ComputationalConfig = ComputationalConfig(),
        metrics: Optional[Sequence[str]] = None,
        resume: bool = False,
    ) -> None:
        """ """
        self.maps.read()
        self._create_split(split.index)

        self._seed(computational)

        if metrics:
            metrics_handler = self._metrics.subset(metrics)
        else:
            metrics_handler = self._metrics

        self._reset_train(split=split, metrics=metrics_handler)
        self._model_to(computational)

        optimizers = self.model.build_optimizers()
        grad_scaler = computational.get_scaler()

        if resume:
            self._call_event(
                Events.RESUME,
                split=split,
                optimizers=optimizers,
                grad_scaler=grad_scaler,
                optimization=self.optimization,
                metrics=metrics_handler,
                callbacks=self.callbacks,
                computational=computational,
            )
        else:
            self._call_event(
                Events.TRAIN_START,
                split=split,
                optimizers=optimizers,
                optimization=self.optimization,
                metrics=metrics_handler,
                callbacks=self.callbacks,
                computational=computational,
            )

        for epoch in range(
            self.state.current_epoch + 1, self.optimization.num_epochs + 1
        ):
            if self.state.should_stop:
                break

            self._reset_epoch(epoch, train_loader=split.train_loader)

            self._call_event(Events.EPOCH_START)

            for batch_idx, batch in enumerate(split.train_loader, start=1):
                self.state.current_train_batch = batch_idx

                self._call_event(Events.BATCH_START, batch=batch)

                self._batch_to(batch, computational=computational)

                self._call_event(Events.FORWARD_START, batch=batch)

                with autocast(
                    device_type=computational.device.type,
                    enabled=computational.amp,
                ):
                    loss = self.model.forward_step(batch=batch)

                self._call_event(
                    Events.BACKWARD_START, loss=loss, grad_scaler=grad_scaler
                )

                self.model.backward_step(loss, grad_scaler=grad_scaler)

                self._call_event(Events.BACKWARD_END)

                if batch_idx % self.optimization.accumulation_steps == 0:
                    self._call_event(
                        Events.OPTIM_STEP_START,
                        optimizers=optimizers,
                        grad_scaler=grad_scaler,
                    )

                    self.model.optimization_step(grad_scaler=grad_scaler)
                    self.state.optim_step += 1

                    grad_scaler.update()
                    for optimizer in optimizers.values():
                        optimizer.zero_grad(set_to_none=True)

                    self._call_event(
                        Events.OPTIM_STEP_END,
                        optimizers=optimizers,
                        grad_scaler=grad_scaler,
                    )

                self._call_event(Events.BATCH_END)

            if (
                self.state.current_epoch
                - 1
                % self.optimization.evaluation_steps  # always validate the first epoch
                == 0
            ):
                self._validation(
                    split.val_loader,
                    metrics=metrics_handler,
                    computational=computational,
                )

            self._call_event(Events.EPOCH_END)

        self._call_event(Events.TRAIN_END)

    def validate(
        self,
        split_idx: int,
        metrics: Sequence[str],
        dataloader: Optional[DataLoader] = None,
        model_checkpoint: Optional[str] = None,
        computational: ComputationalConfig = ComputationalConfig(),
    ) -> None:
        """
        Evaluate the model on a validation or test dataset.
        """
        self.maps.read()
        self._check_split(split_idx)

        self._seed(computational)

        if not dataloader:
            dataloader = self._get_dataloader(
                self.maps.training.data.validation.splits[split_idx]
            )

        metrics_handler = self._metrics.subset(metrics)

        for chkpt_name, chkpt in self._get_models_in_split(split_idx, model_checkpoint):
            self._reset_validate(split_idx, dataloader, metrics_handler)
            self._load_model_checkpoint(chkpt)
            self._model_to(computational)

            self._call_event(
                Events.VALIDATE_START,
                dataloader=dataloader,
                model_checkpoint=chkpt_name,
                metrics=metrics_handler,
                callbacks=self.callbacks,
                computational=computational,
            )

            self._evaluation_loop(
                dataloader,
                metrics=metrics_handler,
                computational=computational,
            )

            self._call_event(
                Events.VALIDATE_END,
                metrics=metrics_handler,
            )

    def test(
        self,
        model_checkpoint: str,
        metrics: Sequence[str],
        group_name: str,
        dataloader: Optional[DataLoader] = None,
        computational: ComputationalConfig = ComputationalConfig(),
    ) -> None:
        """
        Evaluate the model on a validation or test dataset.
        """
        self.maps.read()

        self._seed(computational)

        if not dataloader:
            self._check_group_exists(group_name)
            dataloader = self._get_dataloader(self.maps.test.groups[group_name])

        metrics_handler = self._metrics.subset(metrics)

        self._reset_test(dataloader, metrics_handler)
        self._load_model_checkpoint(
            self.maps.training.get_checkpoint_dir(model_checkpoint)
        )
        self._model_to(computational)

        self._call_event(
            Events.TEST_START,
            dataloader=dataloader,
            model_checkpoint=model_checkpoint,
            metrics=metrics_handler,
            group_name=group_name,
            callbacks=self.callbacks,
            computational=computational,
        )

        self._evaluation_loop(
            dataloader, metrics=metrics_handler, computational=computational
        )

        self._call_event(
            Events.TEST_END,
            metrics=metrics_handler,
        )

    def _validation(
        self,
        dataloader: DataLoader,
        metrics: MetricsHandler,
        computational: ComputationalConfig,
    ) -> None:
        self._reset_validation(dataloader, metrics)

        self._call_event(
            Events.VAL_START,
            dataloader=dataloader,
            metrics=metrics,
            callbacks=self.callbacks,
            computational=computational,
        )

        self._evaluation_loop(
            dataloader,
            metrics=metrics,
            epoch=self.state.current_epoch,
            computational=computational,
        )

        self._call_event(
            Events.VAL_END,
            metrics=metrics,
        )

    def _evaluation_loop(
        self,
        dataloader: DataLoader,
        metrics: MetricsHandler,
        computational: ComputationalConfig,
        epoch: Optional[int] = None,
    ) -> None:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader, start=1):
                self.state.current_val_batch = batch_idx

                self._call_event(Events.BATCH_START, batch=batch)

                self._batch_to(batch, computational=computational)

                self._call_event(Events.EVAL_START, batch=batch)

                output_batch = self.model.evaluation_step(batch)

                self._call_event(
                    Events.EVAL_END,
                    output=output_batch,
                    detailed_metrics_df=metrics_df,
                )

                metrics_df = metrics(
                    output_batch, epoch=self.state.current_epoch, metrics=metrics
                )

                self._call_event(Events.BATCH_END)

        metrics.aggregate(epoch=epoch)

    def _reset_train(self, split: Split, metrics: MetricsHandler) -> None:
        self.state.reset_training(
            split_idx=split.index, num_epochs=self._optim_config.num_epochs
        )
        self.model.reset()
        split.train_loader.dataset.train()
        split.val_loader.dataset.eval()
        metrics.reset(reset_df=True)

    def _reset_epoch(self, epoch: int, train_loader: DataLoader) -> None:
        self.state.reset_epoch(current_epoch=epoch, train_loader=train_loader)
        self.model.train()
        train_loader.set_epoch(epoch)

    def _reset_validation(
        self, val_loader: DataLoader, metrics: MetricsHandler
    ) -> None:
        self.state.reset_validation(
            split_idx=self.state.split_idx, val_loader=val_loader, in_training=True
        )
        self.model.eval()
        metrics.reset(reset_df=False)

    def _reset_validate(
        self, split_idx: int, dataloader: DataLoader, metrics: MetricsHandler
    ) -> None:
        self.state.reset_validation(
            split_idx=split_idx, val_loader=dataloader, in_training=False
        )
        self.model.eval()
        dataloader.dataset.eval()
        metrics.reset(reset_df=True)

    def _reset_test(self, dataloader: DataLoader, metrics: MetricsHandler) -> None:
        self.state.reset_test(dataloader)
        self.model.eval()
        dataloader.dataset.eval()
        metrics.reset(reset_df=True)

    @staticmethod
    def _seed(computational: ComputationalConfig) -> None:
        if computational.seed is not None:
            seed_everything(
                computational.seed, deterministic=computational.deterministic
            )

    def _model_to(self, comp_config: ComputationalConfig) -> None:
        comp_config.check_device()
        self.model.to(device=comp_config.device, non_blocking=comp_config.non_blocking)
        if comp_config.channels_last:
            try:
                self.model.to(torch.channels_last)
            except RuntimeError:
                self.model.to(torch.channels_last_3d)

    @classmethod
    def _batch_to(cls, batch: BatchType, computational: ComputationalConfig) -> None:
        """
        Send the data to the right device.
        """
        if isinstance(batch, Batch):
            batch.to(
                device=computational.device,
                non_blocking=computational.non_blocking,
                channels_last=computational.channels_last,
            )
        elif isinstance(batch, dict):
            for b in batch.values():
                cls._batch_to(b, computational=computational)
        else:  # batch has been checks by ChecksCallback, so it must be a sequence
            for b in batch:
                cls._batch_to(b, computational=computational)

    def _call_event(self, event: Events, **kwargs):
        self.callbacks.call_event(
            event, model=self.model, maps=self.maps, state=self.state, **kwargs
        )

    def _get_models_in_split(
        self, split_idx: int, model_checkpoint: Optional[str]
    ) -> Iterator[tuple[str, Path]]:
        if model_checkpoint:
            yield (
                model_checkpoint,
                self.maps.training.splits[split_idx]
                .models.get_checkpoint_dir(model_checkpoint)
                .model_pt,
            )
        else:
            for model_checkpoint in self.maps.training.splits[
                split_idx
            ].models.get_all_models():
                yield from self._get_models_in_split(split_idx, model_checkpoint)

    def _load_model_checkpoint(self, model_path: Path) -> None:
        self.model.to(CPU)  # load weights on cpu
        state_dict = self.maps.open_file(model_path)
        self.model.load_state_dict(state_dict)

    def _get_dataloader(self, data_dir: DataDir) -> DataLoader:
        def _error_msg(obj: str, path: Path) -> str:
            return (
                f"ClinicaDL could not read the {obj} in {path}. Please pass directly the dataloader "
                "via 'dataloader'."
            )

        try:
            dataset = get_dataset_from_json(data_dir.dataset_json)
        except Exception as e:
            raise CannotReadJsonError(
                _error_msg("dataset", data_dir.dataset_json)
            ) from e

        try:
            dataloader_config = DataLoaderConfig.from_json(data_dir.dataloader_json)
        except Exception as e:
            raise CannotReadJsonError(
                _error_msg("dataloader", data_dir.dataloader_json)
            ) from e

        return dataloader_config.get_object(dataset)

    def _create_split(self, split_idx: int) -> None:
        if split_idx in self.maps.training.splits_list:
            raise ValueError(
                f"Training on split {split_idx} has already been performed. To relaunch a training on this split, first delete it properly with clinicadl.io.Maps.delete_split"
            )
        self.maps.training.create_split(split_idx)

    def _check_split(self, split_idx: int) -> None:
        if split_idx not in self.maps.training.splits_list:
            raise KeyError(f"No training performed on split {split_idx}.")

    def _check_group_exists(self, group: str) -> None:
        if group not in self.maps.test.groups_list:
            raise ValueError(
                f"The group you passed ('{group}'), so you must pass a dataloader."
            )
