from __future__ import annotations

import shutil
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import pandas as pd
import torch
from monai.metrics.metric import CumulativeIterationMetric as MonaiMetric
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

from clinicadl.callbacks.handler import Callback, CallbacksHandler
from clinicadl.data.dataloader import Batch
from clinicadl.data.datasets import CapsDataset
from clinicadl.dictionary.words import BATCH, EPOCH, LOSS, TIME
from clinicadl.losses.config import LossConfig
from clinicadl.losses.types import Loss
from clinicadl.maps.maps import Maps
from clinicadl.metrics.config import MetricConfig
from clinicadl.metrics.config.enum import Optimum
from clinicadl.metrics.metrics import ClinicaDLMetrics, LossMetricConfig, MetricConfig
from clinicadl.model.clinicadl_model import ClinicaDLModel
from clinicadl.optim.config import OptimizationConfig
from clinicadl.predictor.predictor import Predictor
from clinicadl.splitter.split import Split
from clinicadl.tsvtools.utils import remove_non_empty_dir
from clinicadl.utils.computational.config import ComputationalConfig
from clinicadl.utils.exceptions import ClinicaDLMAPSError
from clinicadl.utils.seed import seed_everything
from clinicadl.utils.typing import PathType


class Trainer:
    def __init__(
        self,
        maps_path: PathType,
        model: ClinicaDLModel,
        callbacks: Optional[list[Callback]] = None,
        metrics: Optional[
            list[Union[MetricConfig, MonaiMetric, LossConfig, Loss]]
        ] = None,
        optim_config: OptimizationConfig = OptimizationConfig(),
        comp_config: ComputationalConfig = ComputationalConfig(),
        _overwrite: bool = True,
        seed: int = 123,
    ) -> None:
        """TO COMPLETE"""

        ## CONFIG
        self.model = model
        self.comp = comp_config
        self.optim = optim_config
        self.metrics = metrics

        self.training_loss = self.init_training_loss()
        self.epoch: int = 0
        self.scaler = self.comp.get_scaler()

        # seed initialization # TODO : check if some arguments can be chosen by the user
        seed_everything(seed, deterministic=False, compensation="memory")

        ## MAPS CONFIG
        self.maps = self.create_maps(maps_path, overwrite=_overwrite)

        # CALLBACKS
        self.callbacks = CallbacksHandler(
            callbacks=callbacks, maps=self.maps, model=self.model
        )
        self.check_metrics()

    def check_metrics(self):
        if self.metrics:
            if not isinstance(self.metrics, list):
                self.metrics = [self.metrics]
            for i, metric in enumerate(self.metrics):
                if isinstance(metric, LossConfig):
                    self.metrics[i] = LossMetricConfig(loss_fn=metric.get_object())
                elif isinstance(metric, Loss):
                    self.metrics[i] = LossMetricConfig(loss_fn=metric)
        else:
            self.metrics = [LossMetricConfig(loss_fn=self.model.loss)]

    @classmethod
    def from_maps(cls, maps_path: PathType) -> Trainer:
        """
        Initialize Trainer from existing MAPS directory.

        Parameters
        ----------
        maps_path : PathType
            Path to the MAPS directory.

        Returns
        -------
        Trainer
            An instance of Trainer initialized with MAPS config.
        """
        maps = Maps(maps_path)
        if not maps.exists():
            raise ValueError(f"Invalid maps file: {maps_path}")

        model = ClinicaDLModel.from_json(maps.model_json)
        metrics = ClinicaDLMetrics.from_json(maps.metrics_json)
        optim = OptimizationConfig.from_json(maps.optimization_json)
        comp = ComputationalConfig.from_json(maps.computational_json)

        return cls(
            maps_path,
            model=model,
            metrics=metrics,
            optim_config=optim,
            comp_config=comp,
            _overwrite=False,
        )

    @classmethod
    def _from_dict(cls, maps_path: PathType, dict_: Dict[str, Any]) -> Trainer:
        """
        Initialize Trainer from a dictionary configuration.

        Parameters
        ----------
        maps_path : PathType
            Path to the MAPS directory.
        dict_ : Dict[str, Any]
            Dictionary containing model, metrics, and config values.

        Returns
        -------
        Trainer
            An instance of Trainer initialized from dictionary.
        """
        model = ClinicaDLModel.from_dict(dict_)
        metrics = ClinicaDLMetrics.from_dict(dict_)
        optim = OptimizationConfig(**dict_)
        comp = ComputationalConfig(**dict_)

        return cls(
            maps_path,
            model=model,
            metrics=metrics,
            optim_config=optim,
            comp_config=comp,
            _overwrite=False,
        )

    def create_maps(self, maps_path: PathType, overwrite: bool) -> Maps:
        """
        Initialize the MAPS folder for saving training results and config files.

        Parameters
        ----------
        maps_path : PathType
            Path to the MAPS directory.
        overwrite : bool
            Whether to overwrite the directory if it exists.
        """
        maps = Maps(maps_path)
        if overwrite:
            if maps.exists():
                remove_non_empty_dir(maps.path)
        else:
            if maps.exists():
                raise ClinicaDLMAPSError(
                    f"The maps directory {maps.path} already exists. Use overwrite=True to remove it."
                )

        maps.create()

        self.model.write_json(maps.model_json)
        self.optim.write_json(maps.optimization_json)
        self.comp.write_json(maps.computational_json)
        self.metrics.write_json(
            maps.metrics_json
        )  # no need to write both train and val metrics

        return maps

    def init_training_loss(self) -> pd.DataFrame:
        """
        Initialize the dataframe to record training loss and time.

        Returns
        -------
        pd.DataFrame
            A dataframe to log loss and computation time per epoch and batch.
        """
        training_loss = pd.DataFrame(columns=[EPOCH, BATCH, TIME, LOSS])
        training_loss.set_index([EPOCH, BATCH], inplace=True)
        training_loss.at[(0, 0), TIME] = 0.0
        training_loss.at[(0, 0), LOSS] = 1.0

        return training_loss

    @property
    def loss(self):
        return self.training_loss[LOSS].iloc[-1]

    def resume(self, split: Split) -> None:
        """
        Resume training from a checkpoint in the MAPS directory.

        Parameters
        ----------
        split : Split
            Split object with dataloaders and split index.
        """

        self.maps.load()

        if split.index not in self.maps.splits:
            raise ClinicaDLMAPSError(
                f"The split {split.index} does not exist in the maps directory."
            )

        self.model.load_optim_state_dict(self.maps.splits[split.index].tmp.optimizer)
        self.epoch = self.model.load_network_state_dict(
            self.maps.splits[split.index].tmp.optimizer
        )
        # TODO: need to resume the lr scheduler and the distributed Sampler
        # TODO: need to load metrics or not ? yes needed

        self.train(split)

    def train(self, split: Split) -> None:
        """
        Train the model on the specified split.

        Parameters
        ----------
        split : Split
            Contains dataloaders for training and validation.
        """
        break_ = False
        self.on_train_begin(split, metrics=metrics)

        while self.epoch < self.optim.epochs:
            if break_:
                break

            self.on_epoch_begin()

            for batch_idx, data in enumerate(split.train_loader):
                self.on_batch_begin(batch_idx=batch_idx)

                with autocast(device_type=self.comp.device.type, enabled=self.comp.amp):
                    loss = self.training_step(data=data)

                self.callbacks.on_backward_begin()

                self.scaler.scale(loss).backward()
                self.weights_update()

                self.on_batch_end(batch_idx=batch_idx, loss=loss)

            self.on_epoch_end(split)
        self.on_train_end(split)

    def on_train_begin(
        self, split: Split, metrics: Optional[list[Union[MetricConfig, MonaiMetric]]]
    ) -> None:
        """
        Initialize components before starting the training loop.

        This includes preparing the MAPS split directory and setting up
        model, optimizer, and data loader for the current split.

        Parameters
        ----------
        split : Split
            The data split (training and validation) used for training.
        """

        self.create_split(split)  # not sure if needed

        self.model.train()

        # self.n_batch = len(split.train_loader)
        # self.n_val_batch = len(split.val_loader)

        self.reset()
        self._init_scheduler(n_batch=len(split.train_loader))

        self.callbacks.on_train_begin(device=self.comp.device.type)

        # self.metrics.on_train_begin()

    def on_epoch_begin(self) -> None:
        """
        Set model and data-related configurations before each training epoch.

        Parameters
        ----------
        dataloader : DataLoader
            The training data loader for the current epoch.
        """
        self.model.network.zero_grad(set_to_none=True)
        self.callbacks.on_epoch_begin(epoch=self.epoch)
        # self.evaluation_flag = True

    def on_batch_begin(self, batch_idx: int):
        """TO COMPLETE"""
        self.callbacks.on_batch_begin(batch=batch_idx)

    def training_step(self, data: Batch) -> torch.Tensor:
        """
        Perform a training step on the model using the provided batch of data and return the computed loss.

        Parameters
        ----------
        data : Batch
            Batch of data including images and labels.

        Returns
        -------
        torch.Tensor
            Computed loss for the batch.
        """
        labels = data.get_labels().to(self.comp.device)
        images = data.get_images().to(self.comp.device)

        outputs = self.model.network(images)
        loss = self.model.loss(outputs, labels)

        return loss

    def weights_update(self):
        """TO COMPLETE"""

        self.scaler.step(self.model.optimizer)
        self.scaler.update()
        self.model.optimizer.zero_grad(set_to_none=True)

    def on_batch_end(self, batch_idx: int, loss: torch.Tensor):
        """TO COMPLETE"""

        self.callbacks.on_batch_end(batch=batch_idx)

        self.training_loss.at[(self.epoch, batch_idx), LOSS] = loss.item()
        self.training_loss.at[(self.epoch, batch_idx), TIME] = self.callbacks.callbacks[
            Chronometer()
        ].time

    def on_epoch_end(self, split: Split) -> None:
        """
        Handle end-of-epoch tasks such as evaluation, saving checkpoints, and logging.

        Parameters
        ----------
        split : Split
            The data split used for training and validation.
        """

        self.validate(split.val_loader)

        self.scheduler.step()  # TODO : to put in callbacks ?

        # Sauvegarde du modèle à la fin de chaque epoch
        self._save_tmp_weights(split.index)  # to put in a callback model_selection

        self.callbacks.on_epoch_end(epoch=self.epoch)
        self.epoch += 1

    def on_train_end(self, split: Split):
        """TO COMPLETE"""

        self.callbacks.on_train_end()

        # self.metrics.on_train_end()
        self.save_metrics(maps=self.maps, split=split.index)

        for name, _ in self.metrics.selection_metrics.items():
            self.model.load_network_state_dict(
                self.maps.splits[split.index].best_metrics[name].model
            )

            validator = Predictor(self.maps.path, self.model, self.comp)
            validator.test(
                split.val_loader,
                metric=name,
                split=split.index,
                data_group="validation",
            )

        self.maps.splits[split.index].tmp.remove()

    def reset(self):
        """TO COMPLETE"""
        self.epoch = 0
        self.metrics.reset(df=True)

    def evaluate(
        self,
        dataloader: DataLoader[CapsDataset],
    ):
        self.callbacks.on_validation_begin()
        self.model.network.eval()
        dataloader.dataset.eval()  # TODO: check that the dataset is a CapsDataset? or do we accept all kind of dataset ?

        self.metrics.reset()

        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                ############
                images = data.get_images().to(self.comp.device)
                labels = data.get_labels().to(self.comp.device)
                ############

                # initialize the loss list to save the loss components
                with autocast(self.comp.device.type, enabled=self.comp.amp):
                    outputs = self.model.network(images)
                    # loss = self.model.loss(outputs, labels)
                    # I think loss is one of callable metrics

                    self.metrics(outputs, labels)
            self.metrics.aggregate(epoch=self.epoch)

        self.model.network.train()

        self.callbacks.on_validation_end()
        return None

    def predict(
        self,
        dataloader: DataLoader[CapsDataset],
        split: int,
        output_transforms: list[Transforms],
        data_group: Optional[str] = None,
    ):
        """TO COMPLETE"""

        validator = Predictor(self.maps.path, self.model, self.comp)
        validator.test(
            dataloader=dataloader,
            additionnal_metrics=[],
            split=split,
            data_group=data_group if data_group else "test",
        )

    ## UTILS

    def save_metrics(self, split: int, maps: Maps):
        """Save the metrics in the MAPS."""
        """Creates a training.tsv file."""

        self.metrics.save(maps.splits[split].best_metrics)

        training_tsv = maps.splits[split].logs.training_tsv
        (training_tsv.parent).mkdir(parents=True, exist_ok=True)
        self.training_loss.to_csv(training_tsv, sep="\t", index=True)

    def create_split(self, split: Split):
        """Check if the split is well defined."""
        if split.train_loader is None:
            raise ValueError(
                "The split has no train_loader defined. Please run `get_dataloader()`"
            )
        if split.val_loader is None:
            raise ValueError(
                "The split has no val_loader defined. Please run `get_dataloader()`"
            )

        self.maps.create_split(split, self.metrics.selection_metrics)
        split.write_json(self.maps.splits[split.index].split_json)

    ## INITIALIZATION
    def _init_scheduler(
        self,
        n_batch: int,
    ):
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.model.optimizer,
            max_lr=self.model.optimizer.param_groups[0]["lr"],
            steps_per_epoch=n_batch,
            epochs=self.optim.epochs,
        )  # TODO: check if it stays ina method init
