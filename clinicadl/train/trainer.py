from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

import torch
from monai.metrics.metric import CumulativeIterationMetric as MonaiMetric
from torch.amp import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

from clinicadl.callbacks.handler import Callback, _CallbacksHandler
from clinicadl.callbacks.training_state import _TrainingState
from clinicadl.data.dataloader import Batch, BatchType
from clinicadl.data.datasets import CapsDataset
from clinicadl.IO.maps.maps import Maps
from clinicadl.losses.config import LossConfig
from clinicadl.losses.types import Loss
from clinicadl.metrics.config import LossMetricConfig, MetricConfig
from clinicadl.metrics.handler import LossMetricConfig, MetricsHandler
from clinicadl.metrics.types import MetricOrConfig
from clinicadl.models import ClinicaDLModel
from clinicadl.optim.config import OptimizationConfig
from clinicadl.predictor.predictor import Predictor
from clinicadl.split.split import Split
from clinicadl.transforms.handlers import Postprocessing, Transforms
from clinicadl.utils.computational.config import ComputationalConfig
from clinicadl.utils.seed import seed_everything
from clinicadl.utils.typing import PathType


class Trainer:
    """
    Trainer class to manage the full lifecycle of model **training**, **evaluation**, and **prediction**
    within the ClinicaDL framework.

    This class encapsulates the training loop, evaluation, and prediction processes while
    integrating callback management, metric tracking, and mixed precision training support.
    It leverages ClinicaDL's components like :py:class:`~clinicadl.models.clinicadl_model.ClinicaDLModel`
    and :py:class:`~clinicadl.IO.maps.maps.Maps`,
    promoting modularity and extensibility primarily through callbacks.

    The Trainer follows a callback-driven design pattern: it invokes callbacks at key stages
    (e.g., training start/end, epoch start/end, batch start/end, backward passes) to enable
    flexible monitoring, logging, early stopping, and other behaviors without modifying
    the core training code.

    .. note:
        This class should generally not be subclassed; custom behavior should be implemented via callbacks.


    Parameters
    ----------
    maps_path : PathType
        Directory path where training outputs, maps, and metrics will be saved.
    model : :py:class:`~clinicadl.models.ClinicaDLModel`
        The deep learning model to train and evaluate.
    callbacks : list[:py:class:`~clinicadl.callbacks.base.Callback`], optional
        List of callback instances to execute during training and evaluation.
        Defaults to None (no callbacks).
    metrics : dict[str, MetricType], optional
        Dictionary of metric names and metric instances for monitoring model performance.
        Defaults to None.
    optim_config : :py:class:`~clinicadl.optim.config.OptimizationConfig`, optional
        Configuration object specifying optimizer settings and training schedule.
        Defaults to `OptimizationConfig()`.
    comp_config : :py:class:`~clinicadl.utils.computational.config.ComputationalConfig`, optional
        Configuration for computation environment (e.g., device type, mixed precision).
        Defaults to `ComputationalConfig()`.
    _overwrite : bool, optional
        Whether to overwrite existing output files in `maps_path`.
        Defaults to False.
    seed : int, optional
        Random seed for reproducibility.
        Defaults to 123.

    Examples
    --------
    .. code-block:: python

        preprocessing_t1 = T1Linear()
        transforms_image = Transforms()

        dataset_t1_image = CapsDataset(
            caps_directory=caps_directory,
            data=sub_ses_t1,
            preprocessing=preprocessing_t1,
            transforms=transforms_image,
            label="diagnosis",
        )
        dataset_t1_image.to_tensors(json_name="test_bis_im.json", n_proc=2)
        splitter = KFold(fold_dir)

        optim_config = OptimizationConfig(epochs=2)
        comp_config = ComputationalConfig(gpu=False)
        dataloader_config = DataLoaderConfig(batch_size=3)

        model = ClinicaDLModel(
            network=get_network_config(
                ImplementedNetwork.RESNET, num_outputs=1, spatial_dims=3, in_channels=1
            ),
            loss = MSELossConfig(),
            optimizer=AdamConfig(),
        )

        metrics = {
            "mae": MAEMetric(),
            "mse": MSEMetricConfig(),
            "matrix": ConfusionMatrixMetricConfig(metric_name=["tpr", "fpr"]
            }

        callbacks = [
            EarlyStopping(metrics=["mae", "loss"]),
            ModelSelection(metrics=["mae"]),
            EarlyStopping(metrics=["mse"]),
            CodeCarbon(),
        ]

        trainer = Trainer(
            maps_path,
            model=model,
            comp_config=comp_config,
            optim_config=optim_config,
            callbacks=callbacks,
            metrics=metrics,
            _overwrite=True,
        )

        for split in splitter.get_splits(dataset=dataset_t1_image):
            split.build_train_loader(dataloader_config)
            split.build_val_loader(dataloader_config)

            trainer.train(split)

    Notes
    -----
    .. note:
        - Training utilizes automatic mixed precision (AMP) if enabled in :py:class:`~clinicadl.utils.computational.config.ComputationalConfig`.
        - The callback system provides hooks to extend training behavior without altering core code.
        - The :py:class:`~clinicadl.train.trainer.Trainer`: expects datasets and models compatible with ClinicaDL interfaces.
        - Metrics can be dynamically updated during evaluation and training.

    """

    def __init__(
        self,
        maps_path: PathType,
        model: ClinicaDLModel,
        callbacks: Optional[list[Callback]] = None,
        metrics: dict[str, MetricOrConfig] = {
            "loss": LossMetricConfig(loss_name="loss")
        },
        optim_config: OptimizationConfig = OptimizationConfig(),
        comp_config: ComputationalConfig = ComputationalConfig(),
        _overwrite: bool = False,
        resume: bool = False,
        seed: int = 123,
    ) -> None:
        maps = Maps(maps_path)
        if not resume:
            train_metrics = MetricsHandler(**metrics)

            self.callbacks = _CallbacksHandler(
                metrics=train_metrics,
                callbacks=callbacks if callbacks is not None else [],
            )
            maps.create(overwrite=_overwrite)

            model.write_json(maps.model_json)
            model.write_architecture_log(maps.architecture_log)

            self.callbacks.write_json(maps.training.callbacks_json)
            train_metrics.write_json(maps.training.metrics_json)
            comp_config.write_json(maps.training.computational_json)
            optim_config.write_json(maps.training.optimization_json)

        else:
            maps.load()
            train_metrics = MetricsHandler.from_json(maps.training.metrics_json)
            self.callbacks = _CallbacksHandler.from_json(maps.training.callbacks_json)
            optim_config = OptimizationConfig.from_json(maps.training.optimization_json)
            comp_config = ComputationalConfig.from_json(
                maps.training.computational_json
            )
            model = ClinicaDLModel.from_json(maps.model_json)

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

    def train(self, split: Split, resume: bool = False) -> None:
        """
        Run the training loop over the given data split.

        Parameters
        ----------
        split : Split
            The data split containing training and validation DataLoaders.
        """
        self.model.train()
        self.model.to(self.config.comp.device)
        split.train_dataset.train()

        # if resume:
        self.on_train_begin(split)

        while not self.config.stop:
            self.on_epoch_begin()
            split.train_loader.set_epoch(self.config.epoch)

            for batch_idx, data in enumerate(split.train_loader):
                self.on_batch_begin(batch_idx=batch_idx)

                self._send_to_device(data)

                with autocast(device_type=self.comp.device.type, enabled=self.comp.amp):
                    loss = self.model.forward_step(data=data, device=self.comp.device)

                self.on_backward_begin()

                self.model.optimization_step(loss, self.scaler)

                self.scaler.update()

                self.on_backward_end()

                self.on_batch_end(loss=loss)

            self.evaluate(split.val_loader)

            self.on_epoch_end(split)

        self.on_train_end(split)

    def evaluate(
        self,
        split: Split,
    ) -> None:
        """
        Evaluate the model on a validation or test dataset.
        """
        self.model.eval()
        self.model.to(self.config.comp.device)
        split.val_dataset.eval()

        self.callbacks.on_validation_begin(config=self.config)

        self.metrics.reset(reset_df=False)

        with torch.no_grad():
            for data in split.val_loader:
                self._send_to_device(data)
                output_batch = self.model.evaluation_step(data)
                self.metrics(output_batch, epoch=self.config.epoch)

        self.metrics.aggregate(epoch=self.config.epoch)

        self.callbacks.on_validation_end(config=self.config)

    def _send_to_device(self, data: BatchType) -> None:
        """
        Send the data to the right device.
        """
        if isinstance(data, Batch):
            data.to(self.config.comp.device)
        else:
            for batch in data:
                batch.to(self.config.comp.device)

    def on_train_begin(self, split: Split) -> None:
        self.reset(split)

        self._write_training_infos(split=split)

        self.callbacks.on_train_begin(config=self.config)

    def on_epoch_begin(self) -> None:
        self.callbacks.on_epoch_begin(config=self.config)

    def on_batch_begin(self, batch_idx: int):
        self.config.batch = batch_idx
        self.callbacks.on_batch_begin(config=self.config)

    def on_backward_begin(self):
        self.callbacks.on_backward_begin(config=self.config)

    def on_backward_end(self):
        self.callbacks.on_backward_end(config=self.config)

    def on_batch_end(self, loss: torch.Tensor):
        self.callbacks.on_batch_end(config=self.config, loss=loss.item())

    def on_epoch_end(self, split: Split) -> None:
        self.callbacks.on_epoch_end(config=self.config)

        if self.config.epoch == self.optim.epochs - 1:
            self.config.stop = True

        self.config.epoch += 1

    def on_train_end(self, split: Split):
        self.callbacks.on_train_end(config=self.config)
        self.metrics.save(self.maps.training.splits[split.index].validation_metrics_tsv)

        self._write_end_training_infos(split=split)

    def reset(self, split: Optional[Split] = None):
        if split:
            self.config.reset(split=split)
        self.metrics.reset(reset_df=True)

    def predict(
        self,
        dataloader: DataLoader[CapsDataset],
        split: int,
        output_transforms: Optional[Union[Transforms, Postprocessing]] = None,
        additional_metrics: Optional[
            list[Union[MetricConfig, MonaiMetric, LossMetricConfig, LossConfig, Loss]]
        ] = None,
        data_group: Optional[str] = None,
    ):
        """
        Predict outputs for a dataset and optionally compute metrics.

        Parameters
        ----------
        dataloader : DataLoader[CapsDataset]
            DataLoader providing the dataset for prediction.
        split : int
            Index of the data split used for prediction.
        output_transforms : Transforms or Postprocessing, optional
            Optional transforms to apply to prediction outputs.
        additional_metrics : list, optional
            Additional metrics or losses to compute during prediction.
        data_group : str, optional
            Group label for the data, e.g., 'test', 'validation'.

        Notes
        -----
        .. note::
            Prediction results and metrics are saved to the configured maps directory.
        """

        # TODO : add transforms to output transforms

        validator = Predictor(self.maps.path, self.model, self.comp)
        validator.test(
            dataloader=dataloader,
            additionnal_metrics=additional_metrics,
            split=split,
            output_transforms=output_transforms,
            data_group=data_group if data_group else "test",
        )

    @classmethod
    def from_maps(cls, maps_path: PathType):
        maps = Maps(maps_path)
        maps.load()

        model = ClinicaDLModel.from_json(maps.model_json)
        comp_config = ComputationalConfig.from_json(maps.training.computational_json)
        optim_config = OptimizationConfig.from_json(maps.training.optimization_json)
        callbacks = _CallbacksHandler.from_json(maps.training.callbacks_json)
        metrics = MetricsHandler.from_json(maps.training.metrics_json)

        # TODO : check seed ?

        return cls(
            maps_path=maps_path,
            model=model,
            callbacks=callbacks,
            metrics=metrics,  # type: ignore
            optim_config=optim_config,
            comp_config=comp_config,
            resume=True,
        )

    def _write_training_infos(
        self,
        split: Split,
    ) -> None:
        """
        Write training information to the maps directory.

        Parameters
        ----------
        split : Split
            The data split used for training.
        """
        self.maps._create_training_split(split=split)
        self.maps._add_lines_to_summary_log(
            f"Training dataset  : {split.train_dataset.caps_reader.input_directory}"
        )

        assert isinstance(split.train_loader.dataset, CapsDataset)
        split.train_loader.dataset.write_json(
            self.maps.training.splits[split.index].caps_dataset_json, name="train"
        )
        split.train_loader_config.write_json(
            self.maps.training.splits[split.index].dataloader_json, name="train"
        )

        assert isinstance(split.val_loader.dataset, CapsDataset)
        split.val_loader.dataset.write_json(
            self.maps.training.splits[split.index].caps_dataset_json, name="val"
        )
        split.val_loader_config.write_json(
            self.maps.training.splits[split.index].dataloader_json, name="val"
        )

    def _write_end_training_infos(
        self,
        split: Split,
    ) -> None:
        """
        Write end of training information to the maps directory.

        Parameters
        ----------
        split : Split
            The data split used for training.
        """

        self.maps._add_lines_to_summary_log(
            f"Input size        : {self.model._input_size}\n"
        )
        self.maps._add_lines_to_summary_log("=" * 15)

        self.config.write_torchsummary()  # not working i don't know why
