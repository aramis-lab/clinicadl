from abc import ABC, abstractmethod
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic.types import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt

from clinicadl.utils.preprocessing import read_preprocessing

from .available_parameters import (
    Compensation,
    ExperimentTracking,
    Mode,
    Optimizer,
    Sampler,
    SizeReductionFactor,
    Transform,
)

logger = getLogger("clinicadl.training_config")


class Task(str, Enum):
    """Tasks that can be performed in ClinicaDL."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    RECONSTRUCTION = "reconstruction"


class CallbacksConfig(BaseModel):
    """Config class to add callbacks to the training."""

    emissions_calculator: bool = False
    track_exp: Optional[ExperimentTracking] = None
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)


class ComputationalConfig(BaseModel):
    """Config class to handle computational parameters."""

    amp: bool = False
    fully_sharded_data_parallel: bool = False
    gpu: bool = True
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)


class CrossValidationConfig(
    BaseModel
):  # TODO : put in data/cross-validation/splitter module
    """
    Config class to configure the cross validation procedure.

    tsv_directory is an argument that must be passed by the user.
    """

    n_splits: NonNegativeInt = 0
    split: Tuple[NonNegativeInt, ...] = ()
    tsv_directory: Path
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    @field_validator("split", mode="before")
    def validator_split(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v  # TODO : check that split exists (and check coherence with n_splits)


class DataConfig(BaseModel):  # TODO : put in data module
    """Config class to specify the data.

    caps_directory and preprocessing_json are arguments
    that must be passed by the user.
    """

    caps_directory: Path
    baseline: bool = False
    diagnoses: Tuple[str, ...] = ("AD", "CN")
    label: Optional[str] = None
    label_code: Dict[str, int] = {}
    multi_cohort: bool = False
    preprocessing_dict: Optional[Dict[str, Any]] = None
    preprocessing_json: Optional[Path] = None
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    @field_validator("diagnoses", mode="before")
    def validator_diagnoses(cls, v):
        """Transforms a list to a tuple."""
        if isinstance(v, list):
            return tuple(v)
        return v  # TODO : check if columns are in tsv

    @model_validator(mode="after")
    def validator_model(self):
        if not self.preprocessing_json and not self.preprocessing_dict:
            raise ValueError("preprocessing_dict or preprocessing_json must be passed.")
        elif self.preprocessing_json:
            read_preprocessing = self.read_json()
            if self.preprocessing_dict:
                assert (
                    read_preprocessing == self.preprocessing_dict
                ), "preprocessings found in preprocessing_dict and preprocessing_json do not match."
            else:
                self.preprocessing_dict = read_preprocessing
        return self

    def read_json(
        self,
    ) -> Dict[str, Any]:  # TODO : create a BaseModel to handle preprocessing?
        """
        Gets the preprocessing dictionary from a preprocessing json file.

        Returns
        -------
        Dict[str, Any]
            The preprocessing dictionary.

        Raises
        ------
        ValueError
            In case of multi-cohort dataset, if no preprocessing file is found in any CAPS.
        """
        from clinicadl.utils.caps_dataset.data import CapsDataset

        if not self.multi_cohort:
            preprocessing_json = (
                self.caps_directory / "tensor_extraction" / self.preprocessing_json
            )
        else:
            caps_dict = CapsDataset.create_caps_dict(
                self.caps_directory, self.multi_cohort
            )
            json_found = False
            for caps_name, caps_path in caps_dict.items():
                preprocessing_json = (
                    caps_path / "tensor_extraction" / self.preprocessing_json
                )
                if preprocessing_json.is_file():
                    logger.info(
                        f"Preprocessing JSON {preprocessing_json} found in CAPS {caps_name}."
                    )
                    json_found = True
            if not json_found:
                raise ValueError(
                    f"Preprocessing JSON {self.preprocessing_json} was not found for any CAPS "
                    f"in {caps_dict}."
                )
        preprocessing_dict = read_preprocessing(preprocessing_json)

        if (
            preprocessing_dict["mode"] == "roi"
            and "roi_background_value" not in preprocessing_dict
        ):
            preprocessing_dict["roi_background_value"] = 0

        return preprocessing_dict

    @computed_field
    @property
    def mode(self) -> Mode:
        return Mode(self.preprocessing_dict["mode"])


class DataLoaderConfig(BaseModel):  # TODO : put in data/splitter module
    """Config class to configure the DataLoader."""

    batch_size: PositiveInt = 8
    n_proc: PositiveInt = 2
    sampler: Sampler = Sampler.RANDOM
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)


class EarlyStoppingConfig(BaseModel):
    """Config class to perform Early Stopping."""

    patience: NonNegativeInt = 0
    tolerance: NonNegativeFloat = 0.0
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)


class LRschedulerConfig(BaseModel):
    """Config class to instantiate an LR Scheduler."""

    adaptive_learning_rate: bool = False
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)


class MAPSManagerConfig(BaseModel):  # TODO : put in model module
    """
    Config class to configure the output MAPS folder.

    output_maps_directory is an argument that must be passed by the user.
    """

    output_maps_directory: Path
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)


class ModelConfig(BaseModel):  # TODO : put in model module
    """
    Abstract config class for the model.

    architecture and loss are specific to the task, thus they need
    to be specified in a subclass.
    """

    architecture: str
    dropout: NonNegativeFloat = 0.0
    loss: str
    multi_network: bool = False
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    @field_validator("dropout")
    def validator_dropout(cls, v):
        assert (
            0 <= v <= 1
        ), f"dropout must be between 0 and 1 but it has been set to {v}."
        return v


class OptimizationConfig(BaseModel):
    """Config class to configure the optimization process."""

    accumulation_steps: PositiveInt = 1
    epochs: PositiveInt = 20
    profiler: bool = False
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)


class OptimizerConfig(BaseModel):
    """Config class to configure the optimizer."""

    learning_rate: PositiveFloat = 1e-4
    optimizer: Optimizer = Optimizer.ADAM
    weight_decay: NonNegativeFloat = 1e-4
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)


class ReproducibilityConfig(BaseModel):
    """Config class to handle reproducibility parameters."""

    compensation: Compensation = Compensation.MEMORY
    deterministic: bool = False
    save_all_models: bool = False
    seed: int = 0
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)


class SSDAConfig(BaseModel):
    """Config class to perform SSDA."""

    caps_target: Path = Path("")
    preprocessing_json_target: Path = Path("")
    ssda_network: bool = False
    tsv_target_lab: Path = Path("")
    tsv_target_unlab: Path = Path("")
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    @computed_field
    @property
    def preprocessing_dict_target(self) -> Dict[str, Any]:  # TODO : check if useful
        """
        Gets the preprocessing dictionary from a target preprocessing json file.

        Returns
        -------
        Dict[str, Any]
            The preprocessing dictionary.
        """
        if not self.ssda_network:
            return {}

        preprocessing_json_target = (
            self.caps_target / "tensor_extraction" / self.preprocessing_json_target
        )

        return read_preprocessing(preprocessing_json_target)


class TransferLearningConfig(BaseModel):
    """Config class to perform Transfer Learning."""

    nb_unfrozen_layer: NonNegativeInt = 0
    transfer_path: Optional[Path] = None
    transfer_selection_metric: str = "loss"
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    @field_validator("transfer_path", mode="before")
    def validator_transfer_path(cls, v):
        """Transforms a False to None."""
        if v is False:
            return None
        return v

    @field_validator("transfer_selection_metric")
    def validator_transfer_selection_metric(cls, v):
        return v  # TODO : check if metric is in transfer MAPS


class TransformsConfig(BaseModel):  # TODO : put in data module?
    """Config class to handle the transformations applied to th data."""

    data_augmentation: Tuple[Transform, ...] = ()
    normalize: bool = True
    size_reduction: bool = False
    size_reduction_factor: SizeReductionFactor = SizeReductionFactor.TWO
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    @field_validator("data_augmentation", mode="before")
    def validator_data_augmentation(cls, v):
        """Transforms lists to tuples and False to empty tuple."""
        if isinstance(v, list):
            return tuple(v)
        if v is False:
            return ()
        return v


class ValidationConfig(BaseModel):
    """
    Abstract config class for the validation procedure.

    selection_metrics is specific to the task, thus it needs
    to be specified in a subclass.
    """

    evaluation_steps: NonNegativeInt = 0
    selection_metrics: Tuple[str, ...]
    valid_longitudinal: bool = False
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)


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
    maps_manager: MAPSManagerConfig
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
