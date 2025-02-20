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
from clinicadl.metrics.metrics import Metrics, TrainingMetrics
from clinicadl.model.clinicadl_model import ClinicaDLModel
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
from clinicadl.utils.seed import seed_everything
from clinicadl.utils.typing import PathType

# from clinicadl.utils.dlo_jz import Chronometer
# from clinicadl.utils.logwriter import LogWriter


logger = getLogger("clinicadl.trainer")


class Trainer:
    def __init__(
        self,
        maps_path: PathType,
    ) -> None:
        """TO COMPLETE"""

        self.reader = MapsReader(maps_path)
        self.maps_path = Path(maps_path)
        self.batch_size: int = 2
        self.epochs: int = 3
        self.lr: float = 0.1
        self.weight_decay: float = 0.0
        self.momentum: float = 0.9
        self.num_workers: int = 0
        self.persistent_workers: bool = True
        self.pin_memory: bool = True
        self.non_blocking: bool = True
        self.prefetch_factor: int = 0
        self.drop_last: bool = False
        self.amp: bool = True
        self.accumulation_steps: int = 1  # gives the number of iterations during which gradients are accumulated before performing the weights update. This allows to virtually increase the size of the batch. Default: 1.
        self.evaluation_steps: int = 5  # gives the number of iterations to perform an evaluation internal to an epoch. Default will only perform an evaluation at the end of each epoch.
        self.current_epoch: int = 0
        self.tolerance = 0
        self.patience = 10
        self.seed = 123
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.deterministic: bool = False
        self.compensation: str = "memory"

        # SEED
        seed_everything(
            self.seed, deterministic=self.deterministic, compensation=self.compensation
        )

        # Chronometer initialisation
        # self.chrono = Chronometer()

        # Initialize the parallel environment
        # dist.init_process_group(backend='nccl', init_method='env://',
        #                         world_size=cluster.size, rank=cluster.rank)

        # define model & device
        # bind the proper GPU to the current process
        # torch.cuda.set_device('cpu')

        # distribute batch size (mini-batch)
        self.num_replica = cluster.size
        self.mini_batch_size = self.batch_size
        self.global_batch_size = self.mini_batch_size * self.num_replica

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

    def train(self, model: ClinicaDLModel, split: Split, metrics: TrainingMetrics):
        """TO COMPLETE"""

        self.validator = Predictor(reader=self.reader, metrics=metrics)

        self.on_train_begin(model, split)

        print(metrics.val_loss.df.iloc[-1].values[0])
        while self.epoch < self.epochs and not self.early_stopping.step(
            metrics.val_loss.df.iloc[-1].values[0]
        ):
            print(f"############# EPOCH {self.epoch}################")
            self.on_epoch_begin(model.network)

            for batch, data in enumerate(split.train_loader):
                print(f"############# BATCH {batch}################")

                ############
                images = (
                    torch.cat(list(i.sample for i in data), dim=0)
                    .unsqueeze(1)
                    .to(self.device)
                )
                labels = (
                    torch.tensor([i.label for i in data], dtype=torch.float32)
                    .unsqueeze(1)
                    .to(self.device)
                )  # TO REMOVE AND CHECK FOR MASK
                ############

                with autocast(self.device.type, enabled=self.amp):
                    outputs = model.network(images)
                    loss = model.loss(outputs, labels)
                    # metrics.compute(
                    #     batch, self.epoch, (outputs, labels), val=False, train= False
                    # ) # compute que le train loss

                metrics.save_loss(loss, batch, self.epoch)

                # loss = metrics.train_loss.get_value(batch= batch, epoch=self.epoch)
                self.scaler.scale(loss).backward()
                self.weights_update(model)

                # # Evaluate the model only when no gradients are accumulated
                # if self.evaluation_steps != 0 and (batch + 1) % self.evaluation_steps == 0:
                #     self.evaluation_flag = False
                #     print(" Evaluate the model only when no gradients are accumulated")
                #     print(f"Évaluation - Epoch {self.epoch}, Batch {batch}:")
                #     self.validator.test(split.val_loader, model, epoch = self.epoch) # compute val metrics + val loss
                del loss

            # PROFILER STEP

            # If no evaluation has been performed, warn the user
            if self.evaluation_flag and self.evaluation_steps != 0:
                print(
                    f"Your evaluation steps {self.evaluation_steps} are too big "
                    f"compared to the size of the dataset. "
                    f"The model is evaluated only once at the end epochs."
                )

            # Update weights one last time if gradients were computed without update
            if (batch + 1) % self.accumulation_steps != 0:
                self.weights_update(model)

            # Always test the results and save them once at the end of the epoch
            model.network.zero_grad(set_to_none=True)
            # self.validator.test(split.train_loader, model, epoch = self.epoch)
            self.validator.test(
                split.val_loader, model, epoch=self.epoch
            )  # compute tout sur val

            self.scheduler.step()  # Update learning rate based on validation loss

            print("increase epoch")
            self.epoch += 1
            # Sauvegarde du modèle à la fin de chaque epoch
            torch.save(
                model.network.state_dict(),
                self.reader.maps_path / f"model_epoch_{self.epoch}.pth",
            )
            # model.network.save_checkpoint(epoch=self.epoch)

            print(metrics.train_loss.df)
            print(metrics.val_loss.df)
            print(metrics.train_metrics.df)
            print(metrics.val_metrics.df)

    def weights_update(self, model: ClinicaDLModel):
        self.scaler.step(model.optimizer)
        self.scaler.update()
        model.optimizer.zero_grad(set_to_none=True)

    def on_epoch_begin(self, network: torch.nn.Module):
        network.zero_grad(set_to_none=True)
        self.evaluation_flag = True

    def on_train_begin(self, model: ClinicaDLModel, split: Split):
        """TO COMPLETE"""

        self.reader._create_maps(overwrite=True)
        self._check_split(split)  # not sure if needed

        model.network.to(self.device)
        model.network.train()

        self.epoch = (
            self.current_epoch
        )  # will be different if resume or transfer learning

        self._init_early_stopping(self.patience, "min", self.tolerance)

        self._init_scaler(self.device, self.amp)

        # profiler = init_profiler(maps_path)

        # TODO: init tracker like WandB or MlFlow (callbacks ?)

        # dataset size 8 x nb de slice
        self.n_batch = len(split.train_loader)

        # Vérification de evaluation_steps
        self._check_evaluation_steps()

        self.n_val_batch = len(split.val_loader)

        self._init_scheduler(model.optimizer, self.lr, self.n_batch, self.epochs)

    def _init_scheduler(
        self,
        optimizer: torch.optim.optimizer.Optimizer,
        lr: float,
        n_batch: int,
        epochs: int,
    ):
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=n_batch, epochs=epochs
        )

    def _init_scaler(
        self,
        device,
        amp,
    ):
        self.scaler = GradScaler(device=device, enabled=amp)

    def _init_early_stopping(self, patience: int, mode: str, tolerance: float):
        config = EarlyStoppingConfig(mode=mode, min_delta=tolerance, patience=patience)
        self.early_stopping = EarlyStopping(config)

    def _check_evaluation_steps(self):
        """Check if the current batch is an evaluation step."""
        # Vérification de evaluation_steps
        if self.evaluation_steps >= self.n_batch:
            print(
                f"Warning: evaluation_steps ({self.evaluation_steps}) >= N_batch ({self.n_batch}) ! Réduction automatique à N_batch // 2."
            )
            self.evaluation_steps = max(
                1, self.n_batch // 2
            )  # Évite d'avoir une valeur trop grande

        elif self.n_batch % self.evaluation_steps != 0:
            print(
                f"Warning: evaluation_steps ({self.evaluation_steps}) ne divise pas exactement N_batch ({self.n_batch})."
            )
            self.evaluation_steps = max(
                1, min(self.evaluation_steps, self.n_batch // 2)
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
