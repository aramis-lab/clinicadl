from __future__ import annotations

import json
import shutil
from logging import getLogger
from pathlib import Path
from typing import Optional, Union

import torch
from monai.metrics.metric import CumulativeIterationMetric as Metric
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel

from clinicadl.maps.maps import Maps
from clinicadl.metrics.config.enum import Optimum
from clinicadl.metrics.config.factory import create_metric_config
from clinicadl.metrics.metrics import Metrics
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
        metrics: Metrics,
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

        # METRICS CONFIG
        metrics.set_loss(model.loss)

        # will be different if resume is called
        self.current_epoch: int = 0

        self.early_stopping = self.optim.init_early_stopping()
        self.scaler = self.comp.init_scaler()

        # seed initialization
        seed_everything(seed, deterministic=False, compensation="memory")

        # Chronometer initialisation
        self.chrono = Chronometer()

        ## MAPS CONFIG
        self.maps = self.init_maps(maps_path, overwrite=_overwrite)

        # Initialize the parallel environment
        # dist.init_process_group(backend='nccl', init_method='env://',
        #                         world_size=cluster.size, rank=cluster.rank)

        # define model & device
        # bind the proper GPU to the current process
        # torch.cuda.set_device('cpu')

        # distribute batch size (mini-batch)
        # self.num_replica = cluster.size
        # self.mini_batch_size = self.batch_size
        # self.global_batch_size = self.mini_batch_size * self.num_replica

        # self.validator = Predictor(self.reader, metrics=me) # need to pass training options

    @classmethod
    def from_maps(cls, maps_path: PathType) -> Trainer:
        """TO COMPLETE"""

        maps = Maps(maps_path)
        if not maps.exists():
            raise ValueError(f"Invalid maps file: {maps_path}")

        model = ClinicaDLModel.from_json(maps.model_json)
        metrics = Metrics.from_json(maps.metrics_json)
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
        metrics = Metrics.from_dict(dict_)
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

    def init_maps(self, maps_path: PathType, overwrite: bool) -> Maps:
        """TO COMPLETE"""
        maps = Maps(maps_path)
        if overwrite:
            if maps.exists():
                remove_non_empty_dir(maps.path)
        else:
            if maps.exists():
                raise ClinicaDLMAPSError(
                    f"The maps directory {maps.path} already exists. Use overwrite=True to remove it."
                )
        return maps

    def write_infos(self):
        self.maps.create()
        self.model.write_json(self.maps.model_json)
        self.optim.write_json(self.maps.optimization_json)
        self.comp.write_json(self.maps.computational_json)
        self.metrics.write_json(self.maps.metrics_json)

    def resume(self, split: Split):
        """TO COMPLETE"""

        self.maps.load()

        if split.index not in self.maps.splits:
            raise ClinicaDLMAPSError(
                f"The split {split.index} does not exist in the maps directory."
            )

        self.model.load_optim_state_dict(self.maps.splits[split.index].tmp.optimizer)
        self.current_epoch = self.model.load_network_state_dict(
            self.maps.splits[split.index].tmp.optimizer
        )
        # TODO: need to resume the lr scheduler and the distributed Sampler
        # metrics = self.reader.load_metrics()

        self.train(split)

    def train(self, split: Split):
        """TO COMPLETE"""

        self.write_infos()
        self.on_train_begin(split)

        while self.epoch < self.optim.epochs and not self.early_stopping.step(
            self.metrics.val.get_loss()
        ):
            self.on_epoch_begin()

            for batch_idx, data in enumerate(split.train_loader):
                self.on_batch_begin()

                with autocast(device_type=self.comp.device.type, enabled=self.comp.amp):
                    loss = self.model.training_step(data=data, device=self.comp.device)

                self.on_batch_end(batch_idx=batch_idx, loss=loss)

            self.on_epoch_end(split)

        self.on_train_end(split)

    def on_train_begin(self, split: Split):
        """TO COMPLETE"""

        self.create_split(split)  # not sure if needed
        self.model.train()

        self.epoch = (
            self.current_epoch
        )  # will be different if resume or transfer learning

        # profiler = init_profiler(maps_path)
        # TODO: init tracker like WandB or MlFlow (callbacks ?)

        self.n_batch = len(split.train_loader)
        self.n_val_batch = len(split.val_loader)

        self._init_scheduler()
        self.chrono.start()

    def on_epoch_begin(self):
        self.model.network.zero_grad(set_to_none=True)
        # self.evaluation_flag = True

    def on_batch_begin(self):
        pass

    def weights_update(self):
        self.scaler.step(self.model.optimizer)
        self.scaler.update()
        self.model.optimizer.zero_grad(set_to_none=True)

    def on_batch_end(self, batch_idx: int, loss: torch.Tensor):
        self.metrics.write_training_loss(
            epoch=self.epoch, batch=batch_idx, loss=loss.item()
        )
        self.scaler.scale(loss).backward()
        self.weights_update()

    def on_epoch_end(self, split: Split):
        # self.model.network.zero_grad(set_to_none=True)
        # Update learning rate based on validation loss

        # PRedictor is initialized here because it depends on the new model
        validator = Predictor(self.maps.path, self.model, self.comp)
        validator.validate(split.val_loader, metrics=self.metrics.val, epoch=self.epoch)

        if self.metrics.compute_train_metrics:
            validator.validate(
                split.train_loader, metrics=self.metrics.train, epoch=self.epoch
            )

        self.scheduler.step()

        # Sauvegarde du modèle à la fin de chaque epoch
        self._save_tmp_weights(split.index)

        self.epoch += 1
        # profiler.step()  # TODO: check this

    def on_train_end(self, split: Split):
        """TO COMPLETE"""
        # profiler.stop()  # TODO: check this

        # TODO: stop tracker like WandB or MlFlow (callbacks ?)

        # self.metrics.on_train_end()
        self.save_metrics(maps=self.maps, split=split.index)

        for metric in self.metrics.val.selection_metrics:
            metric = metric.value

            self.model.load_network_state_dict(
                self.maps.splits[split.index].best_metrics[metric].model
            )

            validator = Predictor(self.maps.path, self.model, self.comp)
            validator.test(
                split.val_loader,
                metric=metric,
                split=split.index,
                data_group="validation",
            )

            if self.metrics.compute_train_metrics:
                validator.test(
                    split.train_loader,
                    metric=metric,
                    split=split.index,
                    data_group="train",
                )

    ## UTILS

    def save_metrics(self, split: int, maps: Maps):
        """Save the metrics in the MAPS."""
        """Creates a training.tsv file."""

        for metric in self.metrics.val.selection_metrics:
            metric = metric.value
            df_to_tsv(
                maps.splits[split].best_metrics[metric].train.metrics_tsv,
                self.metrics.train.df,
            )
            df_to_tsv(
                maps.splits[split].best_metrics[metric].val.metrics_tsv,
                self.metrics.val.df,
            )
        training_tsv = maps.splits[split].logs.training_tsv
        (training_tsv.parent).mkdir(parents=True, exist_ok=True)
        self.metrics.training_loss.to_csv(training_tsv, sep="\t", index=True)

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

        self.maps.create_split(split, self.metrics.val.selection_metrics)
        split.write_json(self.maps.splits[split.index].split_json)

    def _save_tmp_weights(self, split: int):
        model_weights = {
            "model": self.model.network.state_dict(),
            "epoch": self.epoch,
        }
        # optimizer_weights = {
        #         "optimizer": self.model.network.optim_state_dict(optimizer),
        #         "epoch": self.epoch,
        #     }

        checkpoint_path = self.maps.splits[split].tmp.path / "checkpoint.pth.tar"
        torch.save(model_weights, checkpoint_path)

        # optim_checkpoint_path = self.reader.tmp_dir_path(split) / "optimizer.pth.tar"
        # torch.save(optimizer_weights, optim_checkpoint_path)

        for metric_ in self.metrics.train.selection_metrics:
            metric = metric_.value
            metric_path = self.maps.splits[split].best_metrics[metric].path
            metric_path.mkdir(parents=True, exist_ok=True)

            optimum = create_metric_config(metric).optimum()

            if (
                self.epoch == 0
                or (
                    optimum == Optimum.MAX
                    and (
                        self.metrics.val.get_value(self.epoch, metric)
                        > self.metrics.val.get_value(self.epoch - 1, metric)
                    )
                )
                or (
                    optimum == Optimum.MIN
                    and (
                        self.metrics.val.get_value(self.epoch, metric)
                        < self.metrics.val.get_value(self.epoch - 1, metric)
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
