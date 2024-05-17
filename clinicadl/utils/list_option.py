from abc import ABC, abstractmethod
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Annotated, Any, Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, computed_field, field_validator
from pydantic.types import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt

from clinicadl.utils.enum import (
    BIDSModality,
    DTIMeasure,
    DTISpace,
    ExtractionMethod,
    Preprocessing,
    SliceDirection,
    SliceMode,
    SUVRReferenceRegions,
    Tracer,
)


class Test(BaseModel):
    ## Task(str, Enum):
    """Tasks that can be performed in ClinicaDL."""

    CLASSIFICATION = "##ification"
    REGRESSION = "regression"
    RECONSTRUCTION = "reconstruction"

    ## ModalityConfig(BaseModel):
    tsv_file: Optional[Path] = None
    modality: BIDSModality

    ## PETModalityConfig(ModalityConfig):
    tracer: Tracer = Tracer.FFDG
    suvr_reference_region: SUVRReferenceRegions = SUVRReferenceRegions.CEREBELLUMPONS2
    modality: BIDSModality = BIDSModality.PET

    ## CustomModalityConfig(ModalityConfig):
    custom_suffix: str = ""
    modality: BIDSModality = BIDSModality.CUSTOM

    ## DTIModalityConfig(ModalityConfig):
    dti_measure: DTIMeasure = DTIMeasure.FRACTIONAL_ANISOTROPY
    dti_space: DTISpace = DTISpace.ALL
    modality: BIDSModality = BIDSModality.DTI

    ## PreprocessingConfig(BaseModel):
    preprocessing_json: Path
    preprocessing_cls: Preprocessing
    use_uncropped_image: bool = False
    extract_method: ExtractionMethod
    file_type: str  # Optional ??
    save_features: bool = False
    extract_json: Optional[str] = None

    ## PreprocessingImageConfig(PreprocessingConfig):
    extract_method: ExtractionMethod = ExtractionMethod.IMAGE

    ## PreprocessingPatchConfig(PreprocessingConfig):
    patch_size: int = 50
    stride_size: int = 50
    extract_method: ExtractionMethod = ExtractionMethod.PATCH

    ## PreprocessingSliceConfig(PreprocessingConfig):
    slice_direction_cls: SliceDirection = SliceDirection.SAGITTAL
    slice_mode_cls: SliceMode = SliceMode.RGB
    discarded_slices: Annotated[list[PositiveInt], 2] = [0, 0]
    extract_method: ExtractionMethod = ExtractionMethod.SLICE

    ## PreprocessingROIConfig(PreprocessingConfig):
    roi_list: list[str] = []
    roi_uncrop_output: bool = False
    roi_custom_template: str = ""
    roi_custom_pattern: str = ""
    roi_custom_suffix: str = ""
    roi_custom_mask_pattern: str = ""
    roi_background_value: int = 0
    extract_method: ExtractionMethod = ExtractionMethod.ROI

    ## CallbacksConfig
    emissions_calculator: bool = False
    track_exp: Optional[ExperimentTracking] = None

    ## CapsDatasetConfig
    data_df: pd.DataFrame
    transformations: Optional[Callable]
    label_presence: bool
    augmentation_transformations: Optional[Callable] = None
    multi_cohort: bool = False
    caps_dict: Dict
    eval_mode: bool = False

    ## ComputationalConfig
    amp: bool = False
    fully_sharded_data_parallel: bool = False
    gpu: bool = True
    n_proc: PositiveInt = 2

    ## CrossValidationConfig(
    n_splits: NonNegativeInt = 0
    split: Tuple[NonNegativeInt, ...] = ()
    tsv_directory: Path

    ## DataConfig
    caps_directory: Path
    baseline: bool = False
    diagnoses: Tuple[str, ...] = ("AD", "CN")
    label: Optional[str] = None
    label_code: Dict[str, int] = {}
    multi_cohort: bool = False

    ## DataLoaderConfig
    batch_size: PositiveInt = 8
    # n_proc: PositiveInt = 2
    sampler: Sampler = Sampler.RANDOM

    ## EarlyStoppingConfig
    patience: NonNegativeInt = 0
    tolerance: NonNegativeFloat = 0.0

    ## LRschedulerConfig
    adaptive_learning_rate: bool = False

    ## MAPSManagerConfig
    output_maps_directory: Path

    ## ModelConfig
    architecture: str
    dropout: NonNegativeFloat = 0.0  # entre 0 et 1 ??
    loss: str
    multi_network: bool = False

    ## OptimizationConfig
    accumulation_steps: PositiveInt = 1
    epochs: PositiveInt = 20
    profiler: bool = False

    ## OptimizerConfig
    learning_rate: PositiveFloat = 1e-4
    optimizer: Optimizer = Optimizer.ADAM
    weight_decay: NonNegativeFloat = 1e-4

    ## ReproducibilityConfig
    compensation: Compensation = (
        Compensation.MEMORY
    )  # "only when deterministic is True"
    deterministic: bool = False
    save_all_models: bool = False
    seed: int = 0

    ## SSDAConfig
    caps_target: Path = Path("")
    preprocessing_json_target: Path = Path("")
    ssda_network: bool = False
    tsv_target_lab: Path = Path("")
    tsv_target_unlab: Path = Path("")

    ## TransferLearningConfig
    nb_unfrozen_layer: NonNegativeInt = 0
    transfer_path: Optional[Path] = None
    transfer_selection_metric: str = "loss"

    ## TransformsConfig
    data_augmentation: Tuple[Transform, ...] = ()
    normalize: bool = True
    size_reduction: bool = False
    size_reduction_factor: SizeReductionFactor = SizeReductionFactor.TWO

    ## ValidationConfig
    evaluation_steps: NonNegativeInt = 0
    selection_metrics: Tuple[str, ...]
    valid_longitudinal: bool = False

    ## TrainingConfig(BaseModel, ABC)
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

    ## Reconstruction specific
    latent_space_size: PositiveInt = 128
    feature_size: PositiveInt = 1024
    n_conv: PositiveInt = 4
    io_layer_channels: PositiveInt = 8
    recons_weight: PositiveFloat = 1.0
    kl_weight: PositiveFloat = 1.0
    normalization: Normalization = Normalization.BATCH

    ## Classification specific
    selection_threshold: PositiveFloat = 0.0  # Will only be used if num_networks != 1


## option sans place
existing_maps: bool = False
existing_maps: bool = False  # Duplicate entry
maps_path: Path = ""
mode: str = None  # = modality ?
preparation_dl: bool = False
prepare_dl: bool = False  # Duplicate entry
preprocessing: str = None
preprocessing_dict: dict = Field(default_factory=dict)
preprocessing_dict_target: str = ""
preprocessing_json: Path
shuffle: bool = True  # Adding a missing parameter (assuming it's meant to be there
tensorboard: bool = True  # Adding a missing parameter (assuming it's meant to be there)
tsv_path: Path = None
uncropped_roi: bool = False
use_extracted_features: bool = False
validation: Literal["KFoldSplit", "SingleSplit"] = "SingleSplit"
