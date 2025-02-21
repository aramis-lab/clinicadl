from __future__ import annotations

import random
from contextlib import nullcontext
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from monai.metrics.metric import CumulativeIterationMetric as Metric
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from clinicadl.data.datasets.caps_dataset import CapsDataset
from clinicadl.experiment_manager.experiment_manager import ExperimentManager
from clinicadl.experiment_manager.maps_reader import MapsReader
from clinicadl.metrics.metrics import Metrics
from clinicadl.model.clinicadl_model import ClinicaDLModel

# from clinicadl.utils.logwriter import LogWriter
from clinicadl.optim.config import OptimizationConfig
from clinicadl.optim.early_stopping import EarlyStopping, EarlyStoppingConfig
from clinicadl.predictor.predictor import Predictor
from clinicadl.splitter.split import Split

# from clinicadl.utils.cluster.profiler import (
#                 ProfilerActivity,
#                 profile,
#                 schedule,
#                 tensorboard_trace_handler,
#             )
from clinicadl.utils import cluster
from clinicadl.utils.computational.computational import ComputationalConfig
from clinicadl.utils.dlo_jz import Chronometer
from clinicadl.utils.seed import seed_everything
from clinicadl.utils.typing import PathType

logger = getLogger("clinicadl.trainer")


class Trainer:
    def __init__(
        self,
        maps_path: PathType,
        optim_config: OptimizationConfig = OptimizationConfig(),
        comp_config: ComputationalConfig = ComputationalConfig(),
    ) -> None:
        """TO COMPLETE"""

        self.reader = MapsReader(maps_path)
        self.reader._create_maps(overwrite=True)

        #####
        self.maps_path = Path(maps_path)
        self.comp = comp_config
        self.optim = optim_config

        self.current_epoch: int = 0

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

    @classmethod
    def from_json(cls, config_file: Path) -> Trainer:
        """TO COMPLETE"""
        return Trainer()

    @classmethod
    def from_maps(cls, maps_path: str | Path) -> Trainer:
        """TO COMPLETE"""
        return Trainer(maps_path)

    def _init_profiler(self, profiler: bool = True):
        if profiler:
            time = datetime.now().strftime("%H:%M:%S")
            filename = [self.maps_path / "profiler" / f"clinicadl_{time}"]
            dist.broadcast_object_list(filename, src=0)
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=2, warmup=2, active=30, repeat=1),
                on_trace_ready=tensorboard_trace_handler(str(filename)[0]),
                profile_memory=True,
                record_shapes=False,
                with_stack=False,
                with_flops=False,
            )
        else:
            prof = nullcontext()
            prof.step = lambda *args, **kwargs: None  # TODO: check this

        return prof

    def resume(self, split: Split):
        """TO COMPLETE"""

        model = self.reader.get_model()

        model.load_optim_state_dict(
            self.reader.optimizer_path(split.index, resume=True)
        )
        self.current_epoch = model.load_state_dict(
            self.reader.checkpoint_path(split.index, resume=True)
        )
        metrics = self.reader.load_metrics()
        self.train(model, split, metrics)

    def train(self, model: ClinicaDLModel, split: Split, metrics: Metrics):
        """TO COMPLETE"""

        self.validator = Predictor(reader=self.reader, metrics=metrics)
        self.on_train_begin(model, split, metrics)

        while self.epoch < self.optim.epochs and not self.early_stopping.step(
            metrics.val.loss
        ):
            print(f"############# EPOCH {self.epoch}################")
            self.on_epoch_begin(model.network)

            for batch, data in enumerate(split.train_loader):
                print(f"############# BATCH {batch}################")

                self.on_batch_begin()
                ############
                images = (
                    torch.cat(list(i.sample for i in data), dim=0)
                    .unsqueeze(1)
                    .to(self.comp.device)
                )
                labels = (
                    torch.tensor([i.label for i in data], dtype=torch.float32)
                    .unsqueeze(1)
                    .to(self.comp.device)
                )  # TO REMOVE AND CHECK FOR MASK
                ############

                with autocast(self.comp.device.type, enabled=self.comp.amp):
                    outputs = model.network(images)
                    loss = model.loss(outputs, labels)
                    metrics.train.compute(
                        batch, self.epoch, (outputs, labels), loss=loss
                    )

                self.scaler.scale(loss).backward()
                self.weights_update(model)

                self.on_batch_end()
                del loss

            # PROFILER STEP
            # Always test the results and save them once at the end of the epoch
            self.on_epoch_end(split, metrics, model)
            # model.network.save_checkpoint(epoch=self.epoch)

        self.on_train_end(split, metrics)

    def weights_update(self, model: ClinicaDLModel):
        self.scaler.step(model.optimizer)
        self.scaler.update()
        model.optimizer.zero_grad(set_to_none=True)

    def on_batch_begin(self):
        pass

    def on_batch_end(self):
        pass

    def on_epoch_begin(self, network: torch.nn.Module):
        network.zero_grad(set_to_none=True)
        # self.evaluation_flag = True

    def on_epoch_end(self, split: Split, metrics: Metrics, model: ClinicaDLModel):
        model.network.zero_grad(set_to_none=True)
        # Update learning rate based on validation loss
        self.validator.test(
            split.val_loader, model, epoch=self.epoch
        )  # compute tout sur val
        self.scheduler.step()

        # Sauvegarde du modèle à la fin de chaque epoch

        self.reader._write_weights(
            model.network.state_dict(), split.index, metrics, epoch=self.epoch
        )
        # torch.save(
        #     model.network.state_dict(),
        #     self.reader.maps_path / f"model_epoch_{self.epoch}.pth",
        # )

        metrics.on_epoch_end(self.epoch)  # compute mean metrics

        self.epoch += 1
        # profiler.step()  # TODO: check this

    def on_train_begin(self, model: ClinicaDLModel, split: Split, metrics: Metrics):
        """TO COMPLETE"""

        self._check_split(split)  # not sure if needed
        self.reader.init_split(split, metrics)

        model.train()

        self.epoch = (
            self.current_epoch
        )  # will be different if resume or transfer learning

        self._init_early_stopping()
        self._init_scaler()

        # profiler = init_profiler(maps_path)

        # TODO: init tracker like WandB or MlFlow (callbacks ?)

        # dataset size 8 x nb de slice
        self.n_batch = len(split.train_loader)

        # Vérification de evaluation_steps
        # self._check_evaluation_steps()

        self.n_val_batch = len(split.val_loader)

        self._init_scheduler(model.optimizer)

    def on_train_end(self, split: Split, metrics: Metrics):
        """TO COMPLETE"""
        # profiler.stop()  # TODO: check this

        # TODO: stop tracker like WandB or MlFlow (callbacks ?)

        self.reader.save_metrics(
            split, metrics
        )  # maybe put save metrics in metrics instead of reader ?

    def _init_scheduler(
        self,
        optimizer: torch.optim.optimizer.Optimizer,
    ):
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"],
            steps_per_epoch=self.n_batch,
            epochs=self.optim.epochs,
        )

    def _init_scaler(
        self,
    ):
        self.scaler = GradScaler(device=self.comp.device.type, enabled=self.comp.amp)

    def _init_early_stopping(self):
        config = EarlyStoppingConfig(
            mode=self.optim.early_stopping.mode,
            min_delta=self.optim.early_stopping.min_delta,
            patience=self.optim.early_stopping.patience,
        )
        self.early_stopping = EarlyStopping(config)

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
