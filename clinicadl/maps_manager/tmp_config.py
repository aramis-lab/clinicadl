from abc import ABC, abstractmethod
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torchvision.transforms as torch_transforms
from pydantic import (
    BaseModel,
    ConfigDict,
    NegativeInt,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    computed_field,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from clinicadl.caps_dataset.data import return_dataset
from clinicadl.metrics.metric_module import MetricModule
from clinicadl.splitter.split_utils import find_splits
from clinicadl.trainer.tasks_utils import (
    evaluation_metrics,
    generate_label_code,
    get_default_network,
    output_size,
)
from clinicadl.transforms import transforms
from clinicadl.transforms.config import TransformsConfig
from clinicadl.utils.enum import (
    Compensation,
    ExperimentTracking,
    Mode,
    Optimizer,
    Sampler,
    SizeReductionFactor,
    Task,
    Transform,
)
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
    ClinicaDLTSVError,
)
from clinicadl.utils.iotools.data_utils import check_multi_cohort_tsv, load_data_test
from clinicadl.utils.iotools.utils import read_preprocessing

logger = getLogger("clinicadl.tmp")


class TmpConfig(BaseModel):
    """
    arguments needed : caps_directory, maps_path, loss
    """

    output_size: Optional[int] = None
    n_classes: Optional[int] = None
    network_task: Optional[str] = None
    metrics_module: Optional[MetricModule] = None
    split_name: Optional[str] = None
    selection_threshold: Optional[int] = None
    num_networks: Optional[int] = None
    input_size: Optional[Sequence[int]] = None
    validation: str = "SingleSplit"
    std_amp: Optional[bool] = None
    preprocessing_dict: Optional[dict] = None

    emissions_calculator: bool = False
    track_exp: Optional[ExperimentTracking] = None

    amp: bool = False
    fully_sharded_data_parallel: bool = False
    gpu: bool = True

    n_splits: NonNegativeInt = 0
    split: Optional[Tuple[NonNegativeInt, ...]] = None
    tsv_path: Optional[Path] = None  # not needed in predict ?

    caps_directory: Path
    baseline: bool = False
    diagnoses: Tuple[str, ...] = ("AD", "CN")
    data_df: Optional[pd.DataFrame] = None
    label: Optional[str] = None
    label_code: Optional[Dict[str, int]] = None
    multi_cohort: bool = False
    mask_path: Optional[Path] = None
    preprocessing_json: Optional[Path] = None
    data_tsv: Optional[Path] = None
    n_subjects: int = 300

    batch_size: PositiveInt = 8
    n_proc: PositiveInt = 2
    sampler: Sampler = Sampler.RANDOM

    patience: NonNegativeInt = 0
    tolerance: NonNegativeFloat = 0.0

    adaptive_learning_rate: bool = False

    maps_path: Path
    data_group: Optional[str] = None
    overwrite: bool = False
    save_nifti: bool = False

    architecture: str = "default"
    dropout: NonNegativeFloat = 0.0
    loss: str
    multi_network: bool = False

    accumulation_steps: PositiveInt = 1
    epochs: PositiveInt = 20
    profiler: bool = False

    learning_rate: PositiveFloat = 1e-4
    optimizer: Optimizer = Optimizer.ADAM
    weight_decay: NonNegativeFloat = 1e-4

    compensation: Compensation = Compensation.MEMORY
    deterministic: bool = False
    save_all_models: bool = False
    seed: int = 0
    config_file: Optional[Path] = None

    caps_target: Path = Path("")
    preprocessing_json_target: Path = Path("")
    ssda_network: bool = False
    tsv_target_lab: Path = Path("")
    tsv_target_unlab: Path = Path("")

    nb_unfrozen_layer: NonNegativeInt = 0
    transfer_path: Optional[Path] = None
    transfer_selection_metric: str = "loss"

    data_augmentation: Tuple[Transform, ...] = ()
    train_transformations: Optional[Tuple[Transform, ...]] = None
    normalize: bool = True
    size_reduction: bool = False
    size_reduction_factor: SizeReductionFactor = SizeReductionFactor.TWO

    evaluation_steps: NonNegativeInt = 0
    selection_metrics: Tuple[str, ...] = ()
    valid_longitudinal: bool = False
    skip_leak_check: bool = False

    # pydantic config
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def check_mandatory_args(self) -> Self:
        if self.caps_directory is None:
            raise ClinicaDLArgumentError(
                "caps_directory is a mandatory argument and it's set to None"
            )
        if self.tsv_path is None:
            raise ClinicaDLArgumentError(
                "tsv_path is a mandatory argument and it's set to None"
            )
        if self.preprocessing_dict is None:
            raise ClinicaDLArgumentError(
                "preprocessing_dict is a mandatory argument and it's set to None"
            )
        if self.mode is None:
            raise ClinicaDLArgumentError(
                "mode is a mandatory argument and it's set to None"
            )
        if self.network_task is None:
            raise ClinicaDLArgumentError(
                "network_task is a mandatory argument and it's set to None"
            )
        return self

    def check_args(self):
        transfo_config = TransformsConfig(
            normalize=self.normalize,
            size_reduction=self.size_reduction,
            size_reduction_factor=self.size_reduction_factor,
        )

        if self.network_task == "classification":
            from clinicadl.splitter.split_utils import init_split_manager

            if self.n_splits > 1 and self.validation == "SingleSplit":
                self.validation = "KFoldSplit"

            split_manager = init_split_manager(
                validation=self.validation,
                parameters=self.model_dump(),
                split_list=None,
                caps_target=self.caps_target,
                tsv_target_lab=self.tsv_target_lab,
            )
            train_df = split_manager[0]["train"]
            self.n_classes = output_size(self.network_task, None, train_df, self.label)
            self.metrics_module = MetricModule(
                evaluation_metrics(self.network_task), n_classes=self.n_classes
            )

        elif self.network_task == "regression" or self.network_task == "reconstruction":
            self.metrics_module = MetricModule(
                evaluation_metrics(self.network_task), n_classes=None
            )

        else:
            raise NotImplementedError(
                f"Task {self.network_task} is not implemented in ClinicaDL. "
                f"Please choose between classification, regression and reconstruction."
            )

        if self.architecture == "default":
            self.architecture = get_default_network(self.network_task)

        if (self.label_code is None) or (
            len(self.label_code) == 0
        ):  # Allows to set custom label code in TOML
            self.label_code = generate_label_code(
                self.network_task, train_df, self.label
            )

        full_dataset = return_dataset(
            self.caps_directory,
            train_df,
            self.preprocessing_dict,
            multi_cohort=self.multi_cohort,
            label=self.label,
            label_code=self.label_code,
            transforms_config=transfo_config,
        )
        self.num_networks = full_dataset.elem_per_image
        self.output_size = output_size(
            self.network_task, full_dataset.size, full_dataset.df, self.label
        )
        self.input_size = full_dataset.size

        if self.num_networks < 2 and self.multi_network:
            raise ClinicaDLConfigurationError(
                f"Invalid training configuration: cannot train a multi-network "
                f"framework with only {self.num_networks} element "
                f"per image."
            )
        possible_selection_metrics_set = set(evaluation_metrics(self.network_task)) | {
            "loss"
        }
        if not set(self.selection_metrics).issubset(possible_selection_metrics_set):
            raise ClinicaDLConfigurationError(
                f"Selection metrics {self.selection_metrics} "
                f"must be a subset of metrics used for evaluation "
                f"{possible_selection_metrics_set}."
            )

    @model_validator(mode="after")
    def check_gpu(self) -> Self:
        if self.gpu:
            import torch

            if not torch.cuda.is_available():
                raise ClinicaDLArgumentError(
                    "No GPU is available. To run on CPU, please set gpu to false or add the --no-gpu flag if you use the commandline."
                )
        elif self.amp:
            raise ClinicaDLArgumentError(
                "AMP is designed to work with modern GPUs. Please add the --gpu flag."
            )
        return self

    @field_validator("track_exp", mode="before")
    def check_track_exp(cls, v):
        if v == "":
            return None

    @field_validator("split", "diagnoses", "selection_metrics", mode="before")
    def list_to_tuples(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v  # TODO : check that split exists (and check coherence with n_splits)

    def adapt_cross_val_with_maps_manager_info(
        self, maps_manager
    ):  # maps_manager is of type MapsManager but need to be in a MapsConfig type in the future
        # TEMPORARY
        if not self.split:
            self.split = find_splits(maps_manager.maps_path, maps_manager.split_name)
        logger.debug(f"List of splits {self.split}")

    def create_groupe_df(self):
        group_df = None
        if self.data_tsv is not None and self.data_tsv.is_file():
            group_df = load_data_test(
                self.data_tsv,
                self.diagnoses,
                multi_cohort=self.multi_cohort,
            )
        return group_df

    def is_given_label_code(self, _label: str, _label_code: Union[str, Dict[str, int]]):
        return (
            self.label is not None
            and self.label != ""
            and self.label != _label
            and _label_code == "default"
        )

    def check_label(self, _label: str):
        if not self.label:
            self.label = _label

    @field_validator("data_tsv", mode="before")
    @classmethod
    def check_data_tsv(cls, v) -> Path:
        if v is not None:
            if not isinstance(v, Path):
                v = Path(v)
            if not v.is_file():
                raise ClinicaDLTSVError(
                    "The participants_list you gave is not a file. Please give an existing file."
                )
            if v.stat().st_size == 0:
                raise ClinicaDLTSVError(
                    "The participants_list you gave is empty. Please give a non-empty file."
                )
        return v

    @computed_field
    @property
    def caps_dict(self) -> Dict[str, Path]:
        from clinicadl.utils.iotools.clinica_utils import check_caps_folder

        if self.multi_cohort:
            if self.caps_directory.suffix != ".tsv":
                raise ClinicaDLArgumentError(
                    "If multi_cohort is True, the CAPS_DIRECTORY argument should be a path to a TSV file."
                )
            else:
                caps_df = pd.read_csv(self.caps_directory, sep="\t")
                check_multi_cohort_tsv(caps_df, "CAPS")
                caps_dict = dict()
                for idx in range(len(caps_df)):
                    cohort = caps_df.loc[idx, "cohort"]
                    caps_path = Path(caps_df.at[idx, "path"])
                    check_caps_folder(caps_path)
                    caps_dict[cohort] = caps_path
        else:
            check_caps_folder(self.caps_directory)
            caps_dict = {"single": self.caps_directory}

        return caps_dict

    @model_validator(mode="after")
    def check_preprocessing_dict(self) -> Self:
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
        from clinicadl.caps_dataset.data import CapsDataset

        if self.preprocessing_dict is None:
            if self.preprocessing_json is not None:
                if not self.multi_cohort:
                    preprocessing_json = (
                        self.caps_directory
                        / "tensor_extraction"
                        / self.preprocessing_json
                    )
                else:
                    caps_dict = self.caps_dict
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

                self.preprocessing_dict = read_preprocessing(preprocessing_json)

            if (
                self.preprocessing_dict["mode"] == "roi"
                and "roi_background_value" not in self.preprocessing_dict
            ):
                self.preprocessing_dict["roi_background_value"] = 0

        return self

    @computed_field
    @property
    def mode(self) -> Mode:
        return Mode(self.preprocessing_dict["mode"])

    @field_validator("dropout")
    def validator_dropout(cls, v):
        assert (
            0 <= v <= 1
        ), f"dropout must be between 0 and 1 but it has been set to {v}."
        return v

    @computed_field
    @property
    def preprocessing_dict_target(self) -> Dict[str, Any]:
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

    @field_validator("transfer_path", mode="before")
    def validator_transfer_path(cls, v):
        """Transforms a False to None."""
        if v is False:
            return None
        return v

    @field_validator("transfer_selection_metric")
    def validator_transfer_selection_metric(cls, v):
        return v  # TODO : check if metric is in transfer MAPS

    @field_validator("data_augmentation", mode="before")
    def validator_data_augmentation(cls, v):
        """Transforms lists to tuples and False to empty tuple."""
        if isinstance(v, list):
            return tuple(v)
        if v is False:
            return ()
        return v

    def get_transforms(
        self,
    ) -> Tuple[torch_transforms.Compose, torch_transforms.Compose]:
        """
        Outputs the transformations that will be applied to the dataset

        Args:
            normalize: if True will perform MinMaxNormalization.
            data_augmentation: list of data augmentation performed on the training set.

        Returns:
            transforms to apply in train and evaluation mode / transforms to apply in evaluation mode only.
        """
        augmentation_dict = {
            "Noise": transforms.RandomNoising(sigma=0.1),
            "Erasing": torch_transforms.RandomErasing(),
            "CropPad": transforms.RandomCropPad(10),
            "Smoothing": transforms.RandomSmoothing(),
            "Motion": transforms.RandomMotion((2, 4), (2, 4), 2),
            "Ghosting": transforms.RandomGhosting((4, 10)),
            "Spike": transforms.RandomSpike(1, (1, 3)),
            "BiasField": transforms.RandomBiasField(0.5),
            "RandomBlur": transforms.RandomBlur((0, 2)),
            "RandomSwap": transforms.RandomSwap(15, 100),
            "None": None,
        }

        augmentation_list = []
        transformations_list = []

        if self.data_augmentation:
            augmentation_list.extend(
                [
                    augmentation_dict[augmentation]
                    for augmentation in self.data_augmentation
                ]
            )

        transformations_list.append(transforms.NanRemoval())
        if self.normalize:
            transformations_list.append(transforms.MinMaxNormalization())
        if self.size_reduction:
            transformations_list.append(
                transforms.SizeReduction(self.size_reduction_factor)
            )

        all_transformations = torch_transforms.Compose(transformations_list)
        train_transformations = torch_transforms.Compose(augmentation_list)

        return train_transformations, all_transformations

    def check_output_saving_nifti(self, network_task: str) -> None:
        # Check if task is reconstruction for "save_tensor" and "save_nifti"
        if self.save_nifti and network_task != "reconstruction":
            raise ClinicaDLArgumentError(
                "Cannot save nifti if the network task is not reconstruction. Please remove --save_nifti option."
            )
