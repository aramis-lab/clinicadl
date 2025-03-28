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

from clinicadl.experiment_manager.maps import Maps
from clinicadl.metrics.config.enum import Optimum
from clinicadl.metrics.config.factory import create_metric_config
from clinicadl.metrics.metrics import Metrics
from clinicadl.model.clinicadl_model import ClinicaDLModel
from clinicadl.optim.config import OptimizationConfig
from clinicadl.optim.early_stopping import EarlyStoppingConfig
from clinicadl.predictor.predictor import Predictor
from clinicadl.splitter.split import Split
from clinicadl.tsvtools.utils import df_to_tsv, remove_non_empty_dir
from clinicadl.utils import cluster
from clinicadl.utils.computational.computational import ComputationalConfig
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
    ) -> None:
        """TO COMPLETE"""

        ## MAPS CONFIG
        self.maps = Maps(maps_path)
        self.init_maps(overwrite=_overwrite)

        ## CONFIG
        self.model = model
        self.comp = comp_config
        self.optim = optim_config
        self.metrics = metrics

        # METRICS CONFIG
        metrics.set_loss(model.loss)

        # will be different if resume is called
        self.current_epoch: int = 0

        # seed initialization
        # TODO: let the user choose a random seed
        seed_everything(123, deterministic=False, compensation="memory")

        # Chronometer initialisation
        self.chrono = Chronometer()

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

    def init_maps(self, overwrite: bool):
        """TO COMPLETE"""
        if not self.maps.is_empty() and overwrite:
            remove_non_empty_dir(self.maps.path)
        else:
            raise ClinicaDLMAPSError(
                f"The maps directory {self.maps.path} is not empty."
            )

        if not self.maps.exists() or self.maps.is_empty():
            self.maps.create()
            self.model.write_info(self.maps.model_json)
            self.optim.write_info(self.maps.optimization_json)
            self.comp.write_info(self.maps.computational_json)

    @classmethod
    def from_json(cls, json_file: PathType) -> Trainer:
        """TO COMPLETE"""
        json_file = Path(json_file)
        if not json_file.is_file():
            raise FileNotFoundError(f"The json file {json_file} does not exist.")

        with json_file.open(mode="r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format in the maps file.")

        maps_path = data["maps_path"]

        return cls._from_dict(maps_path, data)

    @classmethod
    def from_maps(cls, maps_path: PathType) -> Trainer:
        """TO COMPLETE"""

        maps = Maps(maps_path)
        if not maps.exists():
            raise ValueError(f"Invalid maps file: {maps_path}")

        dict_ = maps.read_maps()  # TODO : read all json files
        return cls._from_dict(maps_path, dict_)

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

    def resume(self, split: Split):
        """TO COMPLETE"""

        self.model.load_optim_state_dict(self.maps.splits[split.index].tmp.optimizer)
        self.current_epoch = self.model.load_state_dict(
            self.maps.splits[split.index].tmp.optimizer
        )
        # TODO: need to resume the lr scheduler and the distributed Sampler
        # metrics = self.reader.load_metrics()

        self.train(split)

    def train(self, split: Split):
        """TO COMPLETE"""

        self.on_train_begin(split)

        while self.epoch < self.optim.epochs and not self.early_stopping.step(
            self.metrics.val.get_loss()
        ):
            self.on_epoch_begin()

            for batch_idx, data in enumerate(split.train_loader):
                self.on_batch_begin()

                images = data.get_images().to(self.comp.device)
                labels = data.get_labels().to(self.comp.device)

                with autocast(self.comp.device.type, enabled=self.comp.amp):
                    outputs = self.model.network(images)
                    loss = self.model.loss(outputs, labels)

                    self.metrics.write_training_loss(
                        epoch=self.epoch, batch=batch_idx, loss=loss.item()
                    )

                self.scaler.scale(loss).backward()
                self.weights_update()

            self.on_epoch_end(split)

        self.on_train_end(split)

    def weights_update(self):
        self.scaler.step(self.model.optimizer)
        self.scaler.update()
        self.model.optimizer.zero_grad(set_to_none=True)

    def on_train_begin(self, split: Split):
        """TO COMPLETE"""

        self._check_split(split)  # not sure if needed
        self.maps.create_split(split, self.metrics.val.selection_metrics)

        self.model.train()

        self.epoch = (
            self.current_epoch
        )  # will be different if resume or transfer learning

        self.early_stopping = self.optim.init_early_stopping()
        self.scaler = self.comp.init_scaler()

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

    def on_batch_end(self):
        pass

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

        for metric in self.metrics.val.selection_metrics:
            metric = metric.value
            self.metrics.train.df.to_csv(
                self.maps.splits[split.index].best_metrics[metric].train.metrics_tsv,
                sep="\t",
                index=False,
            )
            self.metrics.val.df.to_csv(
                self.maps.splits[split.index].best_metrics[metric].val.metrics_tsv,
                sep="\t",
                index=False,
            )

        self.metrics.save_metrics(self.maps.splits[split.index].logs.training_tsv)

        for metric in self.metrics.val.selection_metrics:
            metric = metric.value

            self.model.load_state_dict(
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
            if self.epoch == 0:
                shutil.copyfile(checkpoint_path, metric_path / "model.pth.tar")
            elif optimum == Optimum.MAX and (
                self.metrics.val.get_value(self.epoch, metric)
                > self.metrics.val.get_value(self.epoch - 1, metric)
            ):
                shutil.copyfile(checkpoint_path, metric_path / "model.pth.tar")
            elif optimum == Optimum.MIN and (
                self.metrics.val.get_value(self.epoch, metric)
                < self.metrics.val.get_value(self.epoch - 1, metric)
            ):
                shutil.copyfile(checkpoint_path, metric_path / "model.pth.tar")

    ## INITIALIZATION
    def _init_validator(self):
        return Predictor(
            maps_path=self.maps.path,
            model=self.model,
            comp_config=self.comp,
        )

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

    def _check_split(self, split: Split):
        """Check if the split is well defined."""
        if split.train_loader is None:
            raise ValueError(
                "The split has no train_loader defined. Please run `get_dataloader()`"
            )
        if split.val_loader is None:
            raise ValueError(
                "The split has no val_loader defined. Please run `get_dataloader()`"
            )
