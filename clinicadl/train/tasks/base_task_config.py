from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, PrivateAttr, field_validator
from pydantic.types import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt

from .available_parameters import (
    Compensation,
    ExperimentTracking,
    Mode,
    Optimizer,
    Sampler,
    SizeReductionFactor,
    Transform,
)

logger = getLogger("clinicadl.base_training_config")


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
    n_proc: PositiveInt = 2
    batch_size: PositiveInt = 8
    evaluation_steps: NonNegativeInt = 0
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
    data_augmentation: Tuple[Transform, ...] = ()
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
    n_splits: NonNegativeInt = 0
    split: Tuple[NonNegativeInt, ...] = ()
    # Optimization
    optimizer: Optimizer = Optimizer.ADAM
    epochs: PositiveInt = 20
    learning_rate: PositiveFloat = 1e-4
    adaptive_learning_rate: bool = False
    weight_decay: NonNegativeFloat = 1e-4
    dropout: NonNegativeFloat = 0.0
    patience: NonNegativeInt = 0
    tolerance: NonNegativeFloat = 0.0
    accumulation_steps: PositiveInt = 1
    profiler: bool = False
    # Transfer Learning
    transfer_path: Optional[Path] = None
    transfer_selection_metric: str = "loss"
    nb_unfrozen_layer: NonNegativeInt = 0
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

    @field_validator("data_augmentation", mode="before")
    def false_to_empty(cls, v):
        if v is False:
            return ()
        return v

    @field_validator("dropout")
    def validator_dropout(cls, v):
        assert (
            0 <= v <= 1
        ), f"dropout must be between 0 and 1 but it has been set to {v}."
        return v

    @field_validator("diagnoses")
    def validator_diagnoses(cls, v):
        return v  # TODO : check if columns are in tsv

    @field_validator("transfer_selection_metric")
    def validator_transfer_selection_metric(cls, v):
        return v  # TODO : check if metric is in transfer MAPS

    @field_validator("split")
    def validator_split(cls, v):
        return v  # TODO : check that split exists (and check coherence with n_splits)
