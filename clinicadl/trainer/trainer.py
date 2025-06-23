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
from clinicadl.split.split import Split
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

        self.callbacks = CallbacksHandler(
            callbacks=callbacks if callbacks is not None else [],
        )

        train_metrics = ClinicaDLMetrics(metrics=metrics, loss=model.loss)

        self.callbacks.check_metrics(train_metrics)

        maps = Maps(maps_path, _overwrite)
        maps.create()

        self.config = _TrainingState(
            maps=maps,
            metrics=train_metrics,
            model=model,
            optim=optim_config,
            comp=comp_config,
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

        self.on_train_begin(split)

        while not self.config.stop:
            self.on_epoch_begin()

            for batch_idx, data in enumerate(split.train_loader):
                self.on_batch_begin(batch_idx=batch_idx)

                with autocast(device_type=self.comp.device.type, enabled=self.comp.amp):
                    loss = self.model.training_step(data=data, device=self.comp.device)

                self.callbacks.on_backward_begin(config=self.config)
                self.scaler.scale(loss).backward()
                self.weights_update()

                self.on_batch_end(loss=loss)

            self.on_epoch_end(split)

        self.on_train_end(split)

    def on_train_begin(self, split: Split) -> None:
        """TO COMPLETE"""

        self.model.train()
        self.reset(split)

        self.callbacks.on_train_begin(config=self.config)

    def on_epoch_begin(self) -> None:
        """TO COMPLETE"""

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

        self.callbacks.on_epoch_end(config=self.config)

        if self.config.epoch == self.optim.epochs - 1:
            self.config.stop = True

        self.config.epoch += 1

    def on_train_end(self, split: Split):
        """TO COMPLETE"""

        self.callbacks.on_train_end(config=self.config)

        self.metrics.save(self.maps.splits[split.index].metrics_tsv)
        self.maps.splits[split.index].tmp.remove()

    def reset(self, split: Optional[Split] = None):
        """TO COMPLETE"""
        if split:
            self.config.reset(split=split)
        self.metrics.reset(df=True)

    def evaluate(
        self,
        dataloader: DataLoader[CapsDataset],
        additional_metrics: Optional[
            list[Union[MetricConfig, MonaiMetric, LossMetricConfig, LossConfig, Loss]]
        ] = None,
    ):
        """TO COMPLETE"""
        self.callbacks.on_validation_begin(config=self.config)
        self.model.network.eval()
        dataloader.dataset.eval()  # TODO: check that the dataset is a CapsDataset? or do we accept all kind of dataset ?

        self.metrics.reset()
        # self.metrics.add_metrics(additional_metrics)

        with torch.no_grad():
            for _, data in enumerate(dataloader):
                self.config.metrics = self.model.validation_step(
                    data=data, device=self.comp.device, metrics=self.metrics
                )

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

        # TODO : add transforms to output transforms

        validator = Predictor(self.maps.path, self.model, self.comp)
        validator.test(
            dataloader=dataloader,
            additionnal_metrics=additional_metrics,
            split=split,
            output_transforms=output_transforms,
            data_group=data_group if data_group else "test",
        )
