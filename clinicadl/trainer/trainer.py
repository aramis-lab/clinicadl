from __future__ import annotations

from typing import Optional, Union

import torch
from monai.metrics.metric import CumulativeIterationMetric as MonaiMetric
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

from clinicadl.callbacks.handler import Callback, CallbacksHandler
from clinicadl.data.dataloader import Batch
from clinicadl.data.datasets import CapsDataset
from clinicadl.losses.config import LossConfig
from clinicadl.losses.types import Loss
from clinicadl.maps.maps import Maps
from clinicadl.metrics.config import MetricConfig
from clinicadl.metrics.metrics import ClinicaDLMetrics, LossMetricConfig
from clinicadl.model.clinicadl_model import ClinicaDLModel
from clinicadl.optim.config import OptimizationConfig
from clinicadl.predictor.predictor import Predictor
from clinicadl.splitter.split import Split
from clinicadl.transforms.output_transforms import OutputTransforms
from clinicadl.transforms.transforms import Transforms
from clinicadl.utils.computational.config import ComputationalConfig
from clinicadl.utils.config.training import _TrainingState
from clinicadl.utils.seed import seed_everything
from clinicadl.utils.typing import PathType


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
        _overwrite: bool = False,
        seed: int = 123,
    ) -> None:
        """TO COMPLETE"""

        train_metrics = ClinicaDLMetrics(metrics=metrics, loss=model.loss)
        maps = Maps(maps_path, _overwrite)
        maps.create()

        self.config = _TrainingState(
            maps=maps,
            metrics=train_metrics,
            model=model,
            optim=optim_config,
            comp=comp_config,
        )
        self.callbacks = CallbacksHandler(
            callbacks=callbacks if callbacks is not None else [],
        )
        self.callbacks.check_metrics(train_metrics)

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

        while self.config.epoch < self.optim.epochs:
            if break_:
                break

            self.on_epoch_begin()

            for batch_idx, data in enumerate(split.train_loader):
                self.on_batch_begin(batch_idx=batch_idx)

                with autocast(device_type=self.comp.device.type, enabled=self.comp.amp):
                    outputs, labels = self.model.training_step(
                        data=data, device=self.comp.device
                    )
                    loss = self.model.loss(outputs, labels)

                self.callbacks.on_backward_begin(config=self.config)

                self.scaler.scale(loss).backward()

                self.weights_update()

                self.on_batch_end(loss=loss)

            self.on_epoch_end(split)

        self.on_train_end(split)

    def on_train_begin(self, split: Split) -> None:
        """TO COMPLETE"""

        self.config.reset(split=split)

        self.model.train()
        self.reset()
        self._init_scheduler(n_batch=len(split.train_loader))

        self.callbacks.on_train_begin(config=self.config)

    def on_epoch_begin(self) -> None:
        """TO COMPLETE"""

        self.model.network.zero_grad(set_to_none=True)
        self.callbacks.on_epoch_begin(config=self.config)

    def on_batch_begin(self, batch_idx: int):
        """TO COMPLETE"""
        self.config.batch = batch_idx
        self.callbacks.on_batch_begin(config=self.config)

    def weights_update(self):
        """TO COMPLETE"""

        self.scaler.step(self.model.optimizer)
        self.scaler.update()
        self.model.optimizer.zero_grad(set_to_none=True)

    def on_batch_end(self, loss: torch.Tensor):
        """TO COMPLETE"""

        self.callbacks.on_batch_end(config=self.config, loss=loss.item())

    def on_epoch_end(self, split: Split) -> None:
        """TO COMPLETE"""

        self.evaluate(split.val_loader)
        self.scheduler.step()  # TODO : to put in callbacks ?

        self.callbacks.on_epoch_end(config=self.config)
        self.config.epoch += 1

    def on_train_end(self, split: Split):
        """TO COMPLETE"""

        self.callbacks.on_train_end(config=self.config)
        self.config.metrics.save(self.maps.splits[split.index].metrics_tsv)
        self.maps.splits[split.index].tmp.remove()

    def reset(self):
        """TO COMPLETE"""
        self.config.epoch = 0
        self.metrics.reset(df=True)

    def evaluate(
        self,
        dataloader: DataLoader[CapsDataset],
    ):
        """TO COMPLETE"""
        self.callbacks.on_validation_begin(config=self.config)
        self.model.network.eval()
        dataloader.dataset.eval()  # TODO: check that the dataset is a CapsDataset? or do we accept all kind of dataset ?

        self.metrics.reset()

        with torch.no_grad():
            for _, data in enumerate(dataloader):
                outputs, labels = self.model.training_step(
                    data=data, device=self.comp.device
                )
                self.metrics(outputs, labels)
            self.metrics.aggregate(epoch=self.config.epoch)

        self.model.network.train()

        self.callbacks.on_validation_end(config=self.config)

    def predict(
        self,
        dataloader: DataLoader[CapsDataset],
        split: int,
        output_transforms: Optional[Union[Transforms, OutputTransforms]] = None,
        additional_metrics: Optional[
            list[Union[MetricConfig, MonaiMetric, LossMetricConfig, LossConfig, Loss]]
        ] = None,
        data_group: Optional[str] = None,
    ):
        """TO COMPLETE"""

        validator = Predictor(self.maps.path, self.model, self.comp)
        validator.test(
            dataloader=dataloader,
            additionnal_metrics=additional_metrics,
            split=split,
            output_transforms=output_transforms,
            data_group=data_group if data_group else "test",
        )

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
