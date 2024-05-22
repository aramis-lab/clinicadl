from abc import ABC, abstractmethod
from logging import getLogger

from pydantic import BaseModel, ConfigDict, computed_field

from clinicadl.config.config import (
    CallbacksConfig,
    ComputationalConfig,
    CrossValidationConfig,
    DataConfig,
    DataLoaderConfig,
    EarlyStoppingConfig,
    LRschedulerConfig,
    MapsManagerConfig,
    ModelConfig,
    OptimizationConfig,
    OptimizerConfig,
    ReproducibilityConfig,
    SSDAConfig,
    TransferLearningConfig,
    TransformsConfig,
    ValidationConfig,
)
from clinicadl.utils.enum import Task

logger = getLogger("clinicadl.training_config")


class TrainingConfig(BaseModel, ABC):
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
    model: ModelConfig
    optimization: OptimizationConfig
    optimizer: OptimizerConfig
