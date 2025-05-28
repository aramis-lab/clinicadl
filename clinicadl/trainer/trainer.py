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

from .config import _TrainingConfig


class Trainer:
    def __init__(
        self,
        maps_path: PathType,
        model: ClinicaDLModel,
        callbacks: Optional[list[Callback]] = None,
        metrics: Optional[
            list[Union[MetricConfig, MonaiMetric, LossMetricConfig, LossConfig, Loss]]
        ] = None,
        optim_config: OptimizationConfig = OptimizationConfig(),
        comp_config: ComputationalConfig = ComputationalConfig(),
        _overwrite: bool = True,
        seed: int = 123,
    ) -> None:
        """TO COMPLETE"""

        _metrics = ClinicaDLMetrics(metrics=metrics, loss=model.loss)
        _maps = Maps(maps_path)

        self.config = _TrainingConfig(
            maps=_maps,
            metrics=_metrics,
            model=model,
            optim=optim_config,
            comp=comp_config,
        )
        self.callbacks = CallbacksHandler(
            maps=_maps,
            model=model,
            metrics=self.metrics,
            callbacks=callbacks if callbacks is not None else [],
        )

        self.scaler = comp_config.get_scaler()
        seed_everything(seed=seed, deterministic=False, compensation="memory")

    @property
    def model(self):
        return self.config.model

    @property
    def optim(self):
        return self.config.optim

    @property
    def comp(self):
        return self.config.comp

    @property
    def metrics(self):
        return self.config.metrics

    @property
    def maps(self):
        return self.config.maps

    def train(self, split: Split) -> None:
        """TO COMPLETE"""

        break_ = False
        self.on_train_begin(split)

        while self.epoch < self.optim.epochs:
            if break_:
                break

            self.on_epoch_begin()

            for batch_idx, data in enumerate(split.train_loader):
                self.on_batch_begin(batch_idx=batch_idx)

                with autocast(device_type=self.comp.device.type, enabled=self.comp.amp):
                    loss = self.training_step(data=data)

                self.callbacks.on_backward_begin(config=self.config)

                self.scaler.scale(loss).backward()
                self.weights_update()

                self.on_batch_end(loss=loss)

            self.on_epoch_end(split)
        self.on_train_end(split)

    def on_train_begin(self, split: Split) -> None:
        """TO COMPLETE"""

        self.config.reset(split=split.index)
        self.create_split(split)  # not sure if needed

        self.model.train()
        self.reset()
        self._init_scheduler(n_batch=len(split.train_loader))

        self.callbacks.on_train_begin(config=self.config, device=self.comp.device.type)

    def on_epoch_begin(self) -> None:
        """TO COMPLETE"""

        self.model.network.zero_grad(set_to_none=True)
        self.config.epoch += 1
        self.callbacks.on_epoch_begin(config=self.config)
        # self.evaluation_flag = True

    def on_batch_begin(self, batch_idx: int):
        """TO COMPLETE"""
        self.config.batch = batch_idx
        self.callbacks.on_batch_begin(config=self.config)

    def training_step(self, data: Batch) -> torch.Tensor:
        """TO COMPLETE"""

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

    def on_batch_end(self, loss: torch.Tensor):
        """TO COMPLETE"""

        self.callbacks.on_batch_end(config=self.config, loss=loss.item())

    def on_epoch_end(self, split: Split) -> None:
        self.evaluate(split.val_loader)

        self.scheduler.step()  # TODO : to put in callbacks ?

        self.callbacks.on_epoch_end(config=self.config)
        self.epoch += 1

    def on_train_end(self, split: Split):
        """TO COMPLETE"""

        self.callbacks.on_train_end(config=self.config)

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
        self.epoch = 0
        self.metrics.reset(df=True)

    def evaluate(
        self,
        dataloader: DataLoader[CapsDataset],
    ):
        self.callbacks.on_validation_begin(config=self.config)
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

        self.callbacks.on_validation_end(config=self.config)
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
        self.metrics.save(maps.splits[split].best_metrics)

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
