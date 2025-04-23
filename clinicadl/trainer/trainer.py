from __future__ import annotations

import json
import shutil
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from monai.metrics.metric import CumulativeIterationMetric as Metric
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from clinicadl.data.dataloader import Batch
from clinicadl.data.datasets import CapsDataset
from clinicadl.maps.maps import Maps
from clinicadl.metrics.config.enum import Optimum
from clinicadl.metrics.config.factory import get_metric_config
from clinicadl.metrics.metrics import ClinicaDLMetrics
from clinicadl.model.clinicadl_model import ClinicaDLModel
from clinicadl.optim.config import OptimizationConfig
from clinicadl.optim.early_stopping import EarlyStoppingConfig
from clinicadl.predictor.predictor import Predictor
from clinicadl.splitter.split import Split
from clinicadl.tsvtools.utils import df_to_tsv, remove_non_empty_dir, tsv_to_df
from clinicadl.utils import cluster
from clinicadl.utils.computational.config import ComputationalConfig
from clinicadl.utils.dlo_jz import Chronometer
from clinicadl.utils.exceptions import ClinicaDLMAPSError
from clinicadl.utils.seed import seed_everything
from clinicadl.utils.typing import PathType

logger = getLogger("clinicadl.trainer")


class Trainer:
    def __init__(
        self,
        maps_path: PathType,
        model: ClinicaDLModel,
        metrics: ClinicaDLMetrics,
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
        self.train_metrics = metrics
        self.metrics = metrics

        # METRICS CONFIG
        self.metrics._init_with_loss(model.loss)
        if self.metrics.compute_train_metrics:
            self.train_metrics = deepcopy(metrics)
            self.train_metrics._init_with_loss(model.loss)

        self.training_loss = self.init_training_loss()

        # will be different if resume is called
        self.epoch: int = 0

        self.early_stopping = self.optim.init_early_stopping()
        self.scaler = self.comp.init_scaler()

        # seed initialization
        seed_everything(seed, deterministic=False, compensation="memory")

        # Chronometer initialisation
        self.chrono = Chronometer()

        ## MAPS CONFIG
        self.init_maps(maps_path, overwrite=_overwrite)

    @classmethod
    def from_maps(cls, maps_path: PathType) -> Trainer:
        """TO COMPLETE"""

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
    def _from_dict(cls, maps_path: PathType, dict_: dict):
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

    def init_maps(self, maps_path: PathType, overwrite: bool):
        """TO COMPLETE"""
        self.maps = Maps(maps_path)
        if overwrite:
            if self.maps.exists():
                remove_non_empty_dir(self.maps.path)
        else:
            if self.maps.exists():
                raise ClinicaDLMAPSError(
                    f"The maps directory {self.maps.path} already exists. Use overwrite=True to remove it."
                )

        self.write_infos()

    def init_training_loss(self) -> pd.DataFrame:
        """TO COMPLETE"""
        training_loss = pd.DataFrame(columns=["epoch", "batch", "time", "Loss"])
        training_loss.set_index(["epoch", "batch"], inplace=True)
        training_loss.at[(0, 0), "time"] = 0.0

        return training_loss

    def write_infos(self):
        """TO COMPLETE"""
        self.maps.create()
        self.model.write_json(self.maps.model_json)
        self.optim.write_json(self.maps.optimization_json)
        self.comp.write_json(self.maps.computational_json)
        self.metrics.write_json(
            self.maps.metrics_json
        )  # no need to write both train and val metrics

    def resume(self, split: Split):
        """TO COMPLETE"""

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
        # TODO: need to load metrics or not ?

        self.train(split)

    def train(self, split: Split):
        """TO COMPLETE"""

        self.on_train_begin(split)
        print("self epoch : ", self.epoch)
        print("self get loss : ", self.metrics.get_loss())

        while self.epoch < self.optim.epochs and not self.early_stopping.step(
            self.metrics.get_loss()
        ):
            self.on_epoch_begin()

            for batch_idx, data in enumerate(split.train_loader):
                self.on_batch_begin()

                with autocast(device_type=self.comp.device.type, enabled=self.comp.amp):
                    loss = self.training_step(data=data)

                self.scaler.scale(loss).backward()
                self.weights_update()

                self.on_batch_end(batch_idx=batch_idx, loss=loss)
            self.on_epoch_end(split)

        self.on_train_end(split)

    def on_train_begin(self, split: Split):
        """TO COMPLETE"""

        self.create_split(split)  # not sure if needed
        self.model.train()

        self.n_batch = len(split.train_loader)
        self.n_val_batch = len(split.val_loader)

        self.reset()
        self._init_scheduler()

        # self.metrics.on_train_begin()

    def reset(self):
        """TO COMPLETE"""
        self.epoch = 0
        self.metrics.reset(df=True)
        self.chrono.start()

    def on_epoch_begin(self):
        """TO COMPLETE"""
        self.model.network.zero_grad(set_to_none=True)
        self.chrono.next_iter()
        # self.evaluation_flag = True

    def on_batch_begin(self):
        """TO COMPLETE"""
        pass

    def training_step(self, data: Batch):
        """
        Perform a training step on the model using the provided batch of data and return the computed loss
        """
        labels = data.get_labels().to(self.comp.device)
        images = data.get_images().to(self.comp.device)

        self.chrono.forward()

        outputs = self.model.network(images)
        loss = self.model.loss(outputs, labels)

        if self.metrics.compute_train_metrics:
            self.train_metrics(outputs, labels)

        return loss

    def weights_update(self):
        """TO COMPLETE"""

        self.chrono.backward()

        self.scaler.step(self.model.optimizer)
        self.scaler.update()
        self.model.optimizer.zero_grad(set_to_none=True)

    def on_batch_end(self, batch_idx: int, loss: torch.Tensor):
        """TO COMPLETE"""

        self.chrono.update()

        if self.metrics.compute_train_metrics:
            self.train_metrics.aggregate(batch=batch_idx, epoch=self.epoch)

        self.training_loss.at[(self.epoch, batch_idx), "Loss"] = loss.item()
        self.training_loss.at[(self.epoch, batch_idx), "time"] = self.chrono.elapsed()

    def on_epoch_end(self, split: Split):
        """TO COMPLETE"""
        # self.model.network.zero_grad(set_to_none=True)
        # Update learning rate based on validation loss

        # PRedictor is initialized here because it depends on the new model

        self.chrono.validation()

        self.validate(split.val_loader)

        self.chrono.validation()

        self.scheduler.step()

        # Sauvegarde du modèle à la fin de chaque epoch
        self._save_tmp_weights(split.index)

        self.epoch += 1
        self.chrono.next_iter()
        # profiler.step()  # TODO: check this

    def on_train_end(self, split: Split):
        """TO COMPLETE"""

        self.chrono.stop()

        # self.metrics.on_train_end()
        self.save_metrics(maps=self.maps, split=split.index)

        for name, metric_config in self.metrics.selection_metrics.items():
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

    def validate(
        self,
        dataloader: DataLoader[CapsDataset],
    ):
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
        return None

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

    def _save_tmp_weights(self, split: int):
        model_weights = {
            "model": self.model.network.state_dict(),
            "epoch": self.epoch,
        }
        checkpoint_path = self.maps.splits[split].tmp.path / "checkpoint.pth.tar"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_weights, checkpoint_path)
        print(split)
        print(self.epoch)
        for name, metric_config in self.metrics.selection_metrics.items():
            metric_path = self.maps.splits[split].best_metrics[name].path
            metric_path.mkdir(parents=True, exist_ok=True)

            optimum = metric_config.optimum()

            if (
                self.epoch == 0
                or (
                    optimum == Optimum.MAX
                    and (
                        self.metrics.get_value(self.epoch, name)
                        > self.metrics.get_value(self.epoch - 1, name)
                    )
                )
                or (
                    optimum == Optimum.MIN
                    and (
                        self.metrics.get_value(self.epoch, name)
                        < self.metrics.get_value(self.epoch - 1, name)
                    )
                )
            ):
                shutil.copyfile(checkpoint_path, metric_path / "model.pth.tar")

    ## INITIALIZATION
    def _init_scheduler(
        self,
    ):
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.model.optimizer,
            max_lr=self.model.optimizer.param_groups[0]["lr"],
            steps_per_epoch=self.n_batch,
            epochs=self.optim.epochs,
        )

    ## CHECK
    def _check_evaluation_steps(self):
        """Check if the current batch is an evaluation step."""
        # Vérification de evaluation_steps
        if self.optim.evaluation_steps >= self.n_batch:
            print(
                f"Warning: evaluation_steps ({self.optim.evaluation_steps}) >= N_batch ({self.n_batch}) ! Réduction automatique à N_batch // 2."
            )
            self.optim.evaluation_steps = max(
                1, self.n_batch // 2
            )  # Évite d'avoir une valeur trop grande

        elif self.n_batch % self.optim.evaluation_steps != 0:
            print(
                f"Warning: evaluation_steps ({self.optim.evaluation_steps}) ne divise pas exactement N_batch ({self.n_batch})."
            )
            self.optim.evaluation_steps = max(
                1, min(self.optim.evaluation_steps, self.n_batch // 2)
            )  # Ajuste pour garder une fréquence raisonnable
