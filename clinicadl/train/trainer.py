from __future__ import annotations

from typing import Optional, Union

import torch
from monai.metrics.metric import CumulativeIterationMetric as MonaiMetric
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

from clinicadl.callbacks.handler import Callback, CallbacksHandler
from clinicadl.callbacks.training_state import _TrainingState
from clinicadl.data.datasets import CapsDataset
from clinicadl.losses.config import LossConfig
from clinicadl.losses.types import Loss
from clinicadl.maps.maps import Maps
from clinicadl.metrics.config import MetricConfig
from clinicadl.metrics.metrics import ClinicaDLMetrics, LossMetricConfig
from clinicadl.metrics.types import MetricType
from clinicadl.model.clinicadl_model import ClinicaDLModel
from clinicadl.optim.config import OptimizationConfig
from clinicadl.predictor.predictor import Predictor
from clinicadl.split.split import Split
from clinicadl.transforms.output_transforms import OutputTransforms
from clinicadl.transforms.transforms import Transforms
from clinicadl.utils.computational.config import ComputationalConfig
from clinicadl.utils.seed import seed_everything
from clinicadl.utils.typing import PathType


class Trainer:
    """
    Trainer class to manage the full lifecycle of model training, evaluation, and prediction
    within the ClinicaDL framework.

    This class encapsulates the training loop, evaluation, and prediction processes while
    integrating callback management, metric tracking, and mixed precision training support.
    It leverages ClinicaDL's components like :py:class:`~clinicadl.model.clinicadl_model.ClinicaDLModel`
    and :py:class:`~clinicadl.maps.maps.Maps`,
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
    model : :py:class:`~clinicadl.model.clinicadl_model.ClinicaDLModel`
        The deep learning model to train and evaluate.
    callbacks : list[Callback], optional
        List of callback instances to execute during training and evaluation.
        Defaults to None (no callbacks).
    metrics : dict[str, MetricType], optional
        Dictionary of metric names and metric instances for monitoring model performance.
        Defaults to None.
    optim_config : OptimizationConfig, optional
        Configuration object specifying optimizer settings and training schedule.
        Defaults to `OptimizationConfig()`.
    comp_config : ComputationalConfig, optional
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
            metrics=,
            _overwrite=True,
        )

        # Cross-validation loop
        for split in splitter.get_splits(dataset=dataset_t1_image):
            split.build_train_loader(dataloader_config)
            split.build_val_loader(dataloader_config)

            trainer.train(split)

    Notes
    -----
    .. note:
        - Training utilizes automatic mixed precision (AMP) if enabled in `comp_config`.
        - The callback system provides hooks to extend training behavior without altering core code.
        - The Trainer expects datasets and models compatible with ClinicaDL interfaces.
        - Metrics can be dynamically updated during evaluation and training.

    """

    def __init__(
        self,
        maps_path: PathType,
        model: ClinicaDLModel,
        callbacks: Optional[list[Callback]] = None,
        metrics: Optional[dict[str, MetricType]] = None,
        optim_config: OptimizationConfig = OptimizationConfig(),
        comp_config: ComputationalConfig = ComputationalConfig(),
        _overwrite: bool = False,
        seed: int = 123,
    ) -> None:
        train_metrics = ClinicaDLMetrics(metrics=metrics, loss=model.loss)

        self.callbacks = CallbacksHandler(
            metrics=train_metrics,
            callbacks=callbacks if callbacks is not None else [],
        )

        self.config = _TrainingState(
            maps=Maps(maps_path, _overwrite),
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
        """
        Run the training loop over the given data split.

        Parameters
        ----------
        split : Split
            The data split containing training and validation DataLoaders.
        """

        self.on_train_begin(split)

        while not self.config.stop:
            self.on_epoch_begin()

            for batch_idx, data in enumerate(split.train_loader):
                self.on_batch_begin(batch_idx=batch_idx)

                with autocast(device_type=self.comp.device.type, enabled=self.comp.amp):
                    loss = self.model.training_step(data=data, device=self.comp.device)

                self.on_backward_begin()
                self.scaler.scale(loss).backward()
                self.on_backward_end()

                self.on_batch_end(loss=loss)

            self.on_epoch_end(split)

        self.on_train_end(split)

    def on_train_begin(self, split: Split) -> None:
        """Prepare training by setting model to training mode, creating maps, and resetting states."""

        self.config.maps.create()
        self.model.train()
        self.reset(split)

        self.callbacks.on_train_begin(config=self.config)

    def on_epoch_begin(self) -> None:
        self.callbacks.on_epoch_begin(config=self.config)

    def on_batch_begin(self, batch_idx: int):
        self.config.batch = batch_idx
        self.callbacks.on_batch_begin(config=self.config)

    def on_backward_begin(self):
        self.callbacks.on_backward_begin(config=self.config)

    def on_backward_end(self):
        self.scaler.step(self.model.optimizer)
        self.scaler.update()
        self.model.optimizer.zero_grad(set_to_none=True)

        self.callbacks.on_backward_end(config=self.config)

    def on_batch_end(self, loss: torch.Tensor):
        self.callbacks.on_batch_end(config=self.config, loss=loss.item())

    def on_epoch_end(self, split: Split) -> None:
        self.evaluate(split.val_loader)

        self.callbacks.on_epoch_end(config=self.config)

        if self.config.epoch == self.optim.epochs - 1:
            self.config.stop = True

        self.config.epoch += 1

    def on_train_end(self, split: Split):
        self.callbacks.on_train_end(config=self.config)
        self.metrics.save(self.maps.splits[split.index].metrics_tsv)

    def reset(self, split: Optional[Split] = None):
        """TO COMPLETE"""
        if split:
            self.config.reset(split=split)
        self.metrics.reset(df=True)

    def evaluate(
        self,
        dataloader: DataLoader[CapsDataset],
        additional_metrics: Optional[list[MetricType]] = None,
    ):
        """
        Evaluate the model on a validation or test dataset.

        Parameters
        ----------
        dataloader : DataLoader[CapsDataset]
            DataLoader providing the dataset to evaluate on.
        additional_metrics : list, optional
            List of additional metrics or losses to compute during evaluation.

        Notes
        -----
        - Evaluation is done in no-grad mode.
        - Model is switched to evaluation mode during the process and reset to train mode after.
        - Metrics are aggregated at the end of evaluation.
        """
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
        """
        Predict outputs for a dataset and optionally compute metrics.

        Parameters
        ----------
        dataloader : DataLoader[CapsDataset]
            DataLoader providing the dataset for prediction.
        split : int
            Index of the data split used for prediction.
        output_transforms : Transforms or OutputTransforms, optional
            Optional transforms to apply to prediction outputs.
        additional_metrics : list, optional
            Additional metrics or losses to compute during prediction.
        data_group : str, optional
            Group label for the data, e.g., 'test', 'validation'.

        Notes
        -----
        .. note:
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
