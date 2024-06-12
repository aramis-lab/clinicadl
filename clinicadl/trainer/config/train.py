from abc import ABC, abstractmethod
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
)

from clinicadl.callbacks.config import CallbacksConfig
from clinicadl.caps_dataset.data_config import DataConfig
from clinicadl.caps_dataset.dataloader_config import DataLoaderConfig
from clinicadl.config.config.computational import ComputationalConfig
from clinicadl.config.config.cross_validation import CrossValidationConfig
from clinicadl.config.config.early_stopping import EarlyStoppingConfig
from clinicadl.config.config.lr_scheduler import LRschedulerConfig
from clinicadl.config.config.maps_manager import MapsManagerConfig
from clinicadl.config.config.reproducibility import ReproducibilityConfig
from clinicadl.config.config.ssda import SSDAConfig
from clinicadl.config.config.transfer_learning import TransferLearningConfig
from clinicadl.config.config.validation import ValidationConfig
from clinicadl.network.config import NetworkConfig
from clinicadl.optimizer.optimization import OptimizationConfig
from clinicadl.optimizer.optimizer import OptimizerConfig
from clinicadl.transforms.config import TransformsConfig
from clinicadl.utils.enum import Task

logger = getLogger("clinicadl.training_config")


class TrainConfig(BaseModel, ABC):
    """

    Abstract config class for the training pipeline.
    Some configurations are specific to the task (e.g. loss function),
    thus they need to be specified in a subclass.
    """

    callbacks: CallbacksConfig
    computational: ComputationalConfig
    cross_validation: CrossValidationConfig
    data: DataConfig
    dataloader: DataLoaderConfig
    early_stopping: EarlyStoppingConfig
    lr_scheduler: LRschedulerConfig
    maps_manager: MapsManagerConfig
    model: NetworkConfig
    optimization: OptimizationConfig
    optimizer: OptimizerConfig
    reproducibility: ReproducibilityConfig
    ssda: SSDAConfig
    transfer_learning: TransferLearningConfig
    transforms: TransformsConfig
    validation: ValidationConfig
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    @computed_field
    @property
    @abstractmethod
    def network_task(self) -> Task:
        """The Deep Learning task to perform."""

    def __init__(self, **kwargs):
        super().__init__(
            callbacks=kwargs,
            computational=kwargs,
            cross_validation=kwargs,
            data=kwargs,
            dataloader=kwargs,
            early_stopping=kwargs,
            lr_scheduler=kwargs,
            maps_manager=kwargs,
            model=kwargs,
            optimization=kwargs,
            optimizer=kwargs,
            reproducibility=kwargs,
            ssda=kwargs,
            transfer_learning=kwargs,
            transforms=kwargs,
            validation=kwargs,
        )

    def _update(self, config_dict: Dict[str, Any]) -> None:
        """Updates the configs with a dict given by the user."""
        self.callbacks.__dict__.update(config_dict)
        self.computational.__dict__.update(config_dict)
        self.cross_validation.__dict__.update(config_dict)
        self.data.__dict__.update(config_dict)
        self.dataloader.__dict__.update(config_dict)
        self.early_stopping.__dict__.update(config_dict)
        self.lr_scheduler.__dict__.update(config_dict)
        self.maps_manager.__dict__.update(config_dict)
        self.model.__dict__.update(config_dict)
        self.optimization.__dict__.update(config_dict)
        self.optimizer.__dict__.update(config_dict)
        self.reproducibility.__dict__.update(config_dict)
        self.ssda.__dict__.update(config_dict)
        self.transfer_learning.__dict__.update(config_dict)
        self.transforms.__dict__.update(config_dict)
        self.validation.__dict__.update(config_dict)

    def update_with_toml(self, path: Union[str, Path]) -> None:
        """
        Updates the configs with a TOML configuration file.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the TOML configuration file.
        """
        from clinicadl.train.utils import extract_config_from_toml_file

        path = Path(path)
        config_dict = extract_config_from_toml_file(path, self.network_task)
        self._update(config_dict)
