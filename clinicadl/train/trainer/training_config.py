from abc import ABC, abstractmethod
from logging import getLogger

from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
)

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
