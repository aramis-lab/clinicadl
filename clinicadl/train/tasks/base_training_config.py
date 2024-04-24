from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, PrivateAttr, field_validator

logger = getLogger("clinicadl.base_training_config")


class Compensation(str, Enum):
    """Available compensations in clinicaDL."""

    MEMORY = "memory"
    TIME = "time"


class SizeReductionFactor(int, Enum):
    """Available size reduction factors in ClinicaDL."""

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5


class ExperimentTracking(str, Enum):
    """Available tools for experiment tracking in ClinicaDL."""

    MLFLOW = "mlflow"
    WANDB = "wandb"


class Sampler(str, Enum):
    """Available samplers in ClinicaDL."""

    RANDOM = "random"
    WEIGHTED = "weighted"


class Mode(str, Enum):
    """Available modes in ClinicaDL."""

    IMAGE = "image"
    PATCH = "patch"
    ROI = "roi"
    SLICE = "slice"


class BaseTaskConfig(BaseModel):
    """
    Base class to handle parameters of the training pipeline.
    """

    caps_directory: Path
    preprocessing_json: Path
    tsv_directory: Path
    output_maps_directory: Path
    # Computational
    gpu: bool = True
    n_proc: int = 2
    batch_size: int = 8
    evaluation_steps: int = 0
    fully_sharded_data_parallel: bool = False
    amp: bool = False
    # Reproducibility
    seed: int = 0
    deterministic: bool = False
    compensation: Compensation = Compensation.MEMORY
    save_all_models: bool = False
    track_exp: Optional[ExperimentTracking] = None
    # Model
    multi_network: bool = False
    ssda_network: bool = False
    # Data
    multi_cohort: bool = False
    diagnoses: Tuple[str, ...] = ("AD", "CN")
    baseline: bool = False
    valid_longitudinal: bool = False
    normalize: bool = True
    data_augmentation: Tuple[str, ...] = ()
    sampler: Sampler = Sampler.RANDOM
    size_reduction: bool = False
    size_reduction_factor: SizeReductionFactor = (
        SizeReductionFactor.TWO
    )  # TODO : change to optional and remove size_reduction parameter
    caps_target: Path = Path("")
    tsv_target_lab: Path = Path("")
    tsv_target_unlab: Path = Path("")
    preprocessing_dict_target: Path = Path(
        ""
    )  ## TODO : change name in commandline. preprocessing_json_target?
    # Cross validation
    n_splits: int = 0
    split: Tuple[int, ...] = ()
    # Optimization
    optimizer: str = "Adam"
    epochs: int = 20
    learning_rate: float = 1e-4
    adaptive_learning_rate: bool = False
    weight_decay: float = 1e-4
    dropout: float = 0.0
    patience: int = 0
    tolerance: float = 0.0
    accumulation_steps: int = 1
    profiler: bool = False
    # Transfer Learning
    transfer_path: Optional[Path] = None
    transfer_selection_metric: str = "loss"
    nb_unfrozen_layer: int = 0
    # Information
    emissions_calculator: bool = False
    # Mode
    use_extracted_features: bool = False  # unused. TODO : remove
    # Private
    _preprocessing_dict: Dict[str, Any] = PrivateAttr()
    _preprocessing_dict_target: Dict[str, Any] = PrivateAttr()
    _mode: Mode = PrivateAttr()

    class ConfigDict:
        validate_assignment = True

    @field_validator("diagnoses", "split", "data_augmentation", mode="before")
    def list_to_tuples(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("transfer_path", mode="before")
    def false_to_none(cls, v):
        if v is False:
            return None
        return v

    @classmethod
    def get_available_optimizers(cls) -> List[str]:
        """To get the list of available optimizers."""
        available_optimizers = [  # TODO : connect to PyTorch to have available optimizers
            "Adadelta",
            "Adagrad",
            "Adam",
            "AdamW",
            "Adamax",
            "ASGD",
            "NAdam",
            "RAdam",
            "RMSprop",
            "SGD",
        ]
        return available_optimizers

    @field_validator("optimizer")
    def validator_optimizer(cls, v):
        available_optimizers = cls.get_available_optimizers()
        assert (
            v in available_optimizers
        ), f"Optimizer '{v}' not supported. Please choose among: {available_optimizers}"
        return v

    @classmethod
    def get_available_transforms(cls) -> List[str]:
        """To get the list of available transforms."""
        available_transforms = [  # TODO : connect to transforms module
            "Noise",
            "Erasing",
            "CropPad",
            "Smoothing",
            "Motion",
            "Ghosting",
            "Spike",
            "BiasField",
            "RandomBlur",
            "RandomSwap",
        ]
        return available_transforms

    @field_validator("data_augmentation", mode="before")
    def validator_data_augmentation(cls, v):
        if v is False:
            return ()

        available_transforms = cls.get_available_transforms()
        for transform in v:
            assert (
                transform in available_transforms
            ), f"Transform '{transform}' not supported. Please pick among: {available_transforms}"
        return v

    @field_validator("dropout")
    def validator_dropout(cls, v):
        assert (
            0 <= v <= 1
        ), f"dropout must be between 0 and 1 but it has been set to {v}."
        return v
