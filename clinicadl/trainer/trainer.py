from __future__ import annotations

from contextlib import nullcontext
from logging import getLogger
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from clinicadl.dataset.datasets.caps_dataset import CapsDataset
from clinicadl.experiment_manager.experiment_manager import ExperimentManager
from clinicadl.metrics.old_metrics.metric_module import RetainBest
from clinicadl.model.clinicadl_model import ClinicaDLModel
from clinicadl.optim.early_stopping import EarlyStopping
from clinicadl.predictor.predictor import Predictor
from clinicadl.splitter.split import Split
from clinicadl.trainer.tasks_utils import get_criterion
from clinicadl.utils import cluster
from clinicadl.utils.logwriter import LogWriter

logger = getLogger("clinicadl.trainer")


class MapsReader:
    maps_path: Path

    def get_model(self) -> ClinicaDLModel:
        return ClinicaDLModel()

    def _write_network_weights(self):
        pass

    def _write_optim_weights(self):
        pass

    def write_tensor(self):
        pass

    def optimizer_path(self, split: int, resume: bool = False) -> Path:
        """TO COMPLETE"""

        checkpoint_path = (
            self.maps_path / f"split-{split}" / "tmp" / "optimizer.pth.tar"
        )
        return checkpoint_path

    def checkpoint_path(self, split: int, resume: bool = False):
        checkpoint_path = (
            self.maps_path / f"split-{split}" / "tmp" / "checkpoint.pth.tar"
        )
        return checkpoint_path


class Trainer:
    def __init__(self, maps_path: Path) -> None:
        """TO COMPLETE"""
        self.reader = MapsReader(maps_path)
        self.maps_path = maps_path

    @classmethod
    def from_json(cls, config_file: Path, manager: ExperimentManager) -> Trainer:
        """TO COMPLETE"""
        return Trainer()

    @classmethod
    def from_maps(cls, maps_path: str | Path) -> Trainer:
        """TO COMPLETE"""
        return Trainer()

    def _init_profiler(self):
        pass

    def resume(self, split: Split):
        """TO COMPLETE"""

        model = self.reader.get_model()

        model.load_optim_state_dict(
            self.reader.optimizer_path(split.index, resume=True)
        )
        current_epoch = model.load_state_dict(
            self.reader.checkpoint_path(split.index, resume=True)
        )

    def train(self, model: ClinicaDLModel, split: Split, epoch: int = 0):
        """TO COMPLETE"""

        # NEEDED ARG
        adaptive_learning_rate: bool = False
        amp: bool = False
        n_epochs: int = 30
        accumulation_steps: int = 3
        evaluation_steps: int = 4
        save_outputs: bool = (
            False  # depend on the network task, only ok for reconstruction
        )
        network_task: str = "classification"  # TASK enum
        #

        # INIT
        criterion = get_criterion(network_task, model.loss)
        early_stopping = EarlyStopping()
        metrics_valid = {"loss": None}
        retain_best = RetainBest()
        scaler = GradScaler("cuda", enabled=amp)
        profiler = self._init_profiler()

        if cluster.master:
            log_writer = LogWriter()

        model.network.train()
        split.train_loader.dataset.train()

        if adaptive_learning_rate:
            from torch.optim.lr_scheduler import ReduceLROnPlateau

            scheduler = ReduceLROnPlateau(model.optimizer, mode="min", factor=0.1)

        validator = Predictor()
        #

        while epoch < n_epochs and not early_stopping.step(metrics_valid["loss"]):
            if isinstance(split.train_loader.sampler, DistributedSampler):
                # It should always be true for a random sampler. But just in case
                # we get a WeightedRandomSampler or a forgotten RandomSampler,
                # we do not want to execute this line.
                split.train_loader.sampler.set_epoch(epoch)

            model.network.zero_grad(set_to_none=True)
            evaluation_flag, step_flag = True, True

            with profiler:
                for i, data in enumerate(split.train_loader):
                    update: bool = (i + 1) % accumulation_steps == 0
                    sync = nullcontext() if update else model.network.no_sync()
                    with sync:
                        with autocast("cuda", enabled=amp):
                            _, loss_dict = model.network(data, criterion)

                        loss = loss_dict["loss"]
                        scaler.scale(loss).backward()

                    if update:
                        step_flag = False
                        scaler.step(model.optimizer)
                        scaler.update()
                        model.optimizer.zero_grad(set_to_none=True)

                        del loss

                        # Evaluate the model only when no gradients are accumulated
                        if evaluation_steps != 0 and (i + 1) % evaluation_steps == 0:
                            evaluation_flag = False

                            _, metrics_train = validator.test(
                                dataloader=split.train_loader
                            )
                            _, metrics_valid = validator.test(
                                dataloader=split.val_loader
                            )

                            model.network.train()
                            split.train_loader.dataset.train()

                            if cluster.master:
                                log_writer.step(
                                    epoch,
                                    i,
                                    metrics_train,
                                    metrics_valid,
                                    len(split.train_loader),
                                )

                    profiler.step()

                # If no step has been performed, raise Exception
                if step_flag:
                    raise ValueError(
                        "The model has not been updated once in the epoch. The accumulation step may be too large."
                    )

                # If no evaluation has been performed, warn the user
                elif evaluation_flag and evaluation_steps != 0:
                    logger.warning(
                        f"Your evaluation steps {evaluation_steps} are too big "
                        f"compared to the size of the dataset. "
                        f"The model is evaluated only once at the end epochs."
                    )

                # Update weights one last time if gradients were computed without update
                if (i + 1) % accumulation_steps != 0:
                    scaler.step(model.optimizer)
                    scaler.update()
                    model.optimizer.zero_grad(set_to_none=True)

                # Always test the results and save them once at the end of the epoch
                model.network.zero_grad(set_to_none=True)
                logger.debug(f"Last checkpoint at the end of the epoch {epoch}")

                _, metrics_train = validator.test(dataloader=split.train_loader)
                _, metrics_valid = validator.test(dataloader=split.val_loader)

                model.network.train()
                split.train_loader.dataset.train()

            if cluster.master:
                # Save checkpoints and best models
                best_dict = retain_best.step(metrics_valid)
                self.reader._write_optim_weights(best_dict)
                self.reader._write_network_weights(best_dict)

            dist.barrier()

            if adaptive_learning_rate:
                scheduler.step(
                    metrics_valid["loss"]
                )  # Update learning rate based on validation loss

            epoch += 1

        del model
        validator._test_loader(dataloader=split.train_loader)
        validator._test_loader(datalaoder=split.val_loader)

        if save_outputs:
            self.reader.write_tensor()
