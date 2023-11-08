import json
import shutil
import subprocess
from contextlib import nullcontext
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
import torch
import torch.distributed as dist
from pydantic import Field, ValidationError
from pydantic.dataclasses import dataclass
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from clinicadl.utils.callbacks.callbacks import Callback, CallbacksHandler
from clinicadl.utils.caps_dataset.data import (
    CapsDataset,
    get_transforms,
    load_data_test,
    return_dataset,
)
from clinicadl.utils.cmdline_utils import check_gpu
from clinicadl.utils.early_stopping import EarlyStopping
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
    ClinicaDLDataLeakageError,
    MAPSError,
)
from clinicadl.utils.logger import setup_logging
from clinicadl.utils.maps_manager.ddp import DDP, cluster, init_ddp
from clinicadl.utils.maps_manager.logwriter import LogWriter
from clinicadl.utils.maps_manager.maps_manager_utils import (
    read_json,
    remove_unused_tasks,
)
from clinicadl.utils.metric_module import RetainBest
from clinicadl.utils.network.network import Network
from clinicadl.utils.seed import get_seed, pl_worker_init_function, seed_everything

logger = getLogger("clinicadl.data_handler")


level_list: List[str] = ["warning", "info", "debug"]
# TODO save weights on CPU for better compatibility


@dataclass
class DataConfig(dict):
    gpu: bool = True
    n_proc: int = 2
    batch_size: int = 8
    evaluation_steps: int = 0
    fully_sharded_data_parallel: bool = False
    amp: bool = False
    seed: int = 0
    deterministic: bool = False
    compensation: str = "memory"  # Only used if deterministic = true
    track_exp: str = ""
    transfer_path: Path = Path("")
    transfer_selection_metric: str = "loss"
    use_extracted_features: bool = False
    n_splits: int = 0
    split: list = Field(default_factory=list)
    optimizer: str = "Adam"
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 0
    tolerance: float = 0.0
    accumulation_steps: int = 1
    profiler: bool = False
    emissions_calculator: bool = False
    validation: Literal["KFoldSplit", "SingleSplit"] = "SingleSplit"
    architecture: str = "default"
    multi_network: bool = False
    ssda_network: bool = False
    dropout: float = Field(0.0, ge=0.0, le=1.0)  # between 0 and 1
    latent_space_size: int = 128
    feature_size: int = 1024
    n_conv: int = 4
    io_layer_channels: int = 8
    recons_weight: int = 1
    kl_weight: int = 1
    normalization: str = "batch"
    network_task: str = "classification"
    selection_metrics: list = Field(default_factory=list)
    label: str = "diagnosis"
    label_code: dict = Field(default_factory=dict)
    selection_threshold: float = 0.0  # Will only be used if num_networks != 1
    loss: str = "CrossEntropyLoss"
    nb_unfrozen_layer: int = 0
    existing_maps: bool = False
    multi_cohort: bool = False
    diagnoses: list = Field(default_factory=list)
    baseline: bool = False
    normalize: bool = True
    data_augmentation: bool = False
    sampler: str = "random"
    size_reduction: bool = False
    size_reduction_factor = 2
    caps_target: Path = ""
    tsv_target_lab: str = ""
    tsv_target_unlab: str = ""
    caps_directory: Path = None
    maps_path: Path = ""
    tsv_path: Path = None
    custom_suffix: str = "custom"
    discarded_slices: int = 1
    extract_json: str = ""
    file_type: str = ""
    mode: str = None
    patch_size: int = 128
    prepare_dl: bool = False
    preprocessing: str = None
    preprocessing_json: str = ""
    preprocessing_dict: dict = Field(default_factory=dict)
    preprocessing_dict_target: str = ""
    roi_custom_mask_pattern: str = ""
    roi_custom_suffix: str = ""
    roi_custom_pattern: str = ""
    roi_custom_template: str = ""
    roi_background_value: int = 0
    roi_list: list = Field(default_factory=list)
    stride_size: int = 2
    slice_direction: str = ""
    slice_mode: str = ""
    suvr_reference_region: str = ""
    tracer: str = "18FFDG"
    uncropped_roi: bool = False
    use_uncropped_image: bool = False
    existing_maps: bool = False

    def __init__(
        self,
        verbose: str = "info",
        task: str = "classification",
        *args,
        **kwargs,
    ):
        ## dict to attribute
        super(DataConfig, self).__init__(**kwargs)
        self.__dict__ = self

        for data in self.__dict__:
            if data in kwargs:
                self.__dict__[data] = kwargs[data]
        self.network_task = task
        print(self.parameters)
        self.parameters = False
        print(self.parameters)

    def from_config_file(self, config_file: Path, task: str):
        """
        Read the configuration file given by the user.
        If it is a TOML file, ensures that the format corresponds to the one in resources.
        Args:
            config_file: path to a configuration file (JSON of TOML).
            task: task learnt by the network (example: classification, regression, reconstruction...).
        Returns:
            dictionary of values ready to use for the MapsManager
        """
        import toml

        from clinicadl.utils.maps_manager.maps_manager_utils import remove_unused_tasks

        if config_file is None:
            raise ClinicaDLConfigurationError("No config file is given")
            # # read default values
            # clinicadl_root_dir = (Path(__file__) / "../..").resolve()
            # config_path = (
            #     Path(clinicadl_root_dir) / "resources" / "config" / "train_config.toml"
            # )
            # config_dict = toml.load(config_path)
            # config_dict = remove_unused_tasks(config_dict, task)
            # config_dict = change_str_to_path(config_dict)

            # # Fill train_dict from TOML files arguments
            # for config_section in config_dict:
            #     for key in config_dict[config_section]:
            #         self.__dict__[key] = config_dict[config_section][key]

        elif config_file.suffix == ".toml":
            user_dict = toml.load(config_file)
            if "Random_Search" in user_dict:
                del user_dict["Random_Search"]

            # read default values
            clinicadl_root_dir = (Path(__file__) / "../../..").resolve()

            print(__file__)
            print(clinicadl_root_dir)
            config_path = (
                Path(clinicadl_root_dir) / "resources" / "config" / "train_config.toml"
            )
            config_dict = toml.load(config_path)
            # Check that TOML file has the same format as the one in clinicadl/resources/config/train_config.toml
            if user_dict is not None:
                for section_name in user_dict:
                    if section_name not in config_dict:
                        raise ClinicaDLConfigurationError(
                            f"{section_name} section is not valid in TOML configuration file. "
                            f"Please see the documentation to see the list of option in TOML configuration file."
                        )
                    for key in user_dict[section_name]:
                        if key not in config_dict[section_name]:
                            raise ClinicaDLConfigurationError(
                                f"{key} option in {section_name} is not valid in TOML configuration file. "
                                f"Please see the documentation to see the list of option in TOML configuration file."
                            )
                        config_dict[section_name][key] = user_dict[section_name][key]

            train_dict = dict()

            # task dependent
            config_dict = remove_unused_tasks(config_dict, task)

            # Fill train_dict from TOML files arguments
            for config_section in config_dict:
                for key in config_dict[config_section]:
                    self.__dict__[key] = config_dict[config_section][key]

        elif config_file.suffix == ".json":
            train_dict = read_json(config_file)
            train_dict = change_str_to_path(train_dict)
            for data in self.__dict__:
                if data in train_dict:
                    self.__dict__[data] = train_dict[data]

        else:
            raise ClinicaDLConfigurationError(
                f"config_file {config_file} should be a TOML or a JSON file."
            )

    def check_existing_maps(self):

        if not (self.maps_path / "maps.json").is_file():
            raise MAPSError(
                f"MAPS was not found at {self.maps_path}."
                f"To initiate a new MAPS please give a train_dict."
            )
        test_parameters = self.get_parameters()
        test_parameters = change_str_to_path(test_parameters)
        parameters = add_default_values(test_parameters)
        for data in self.__dict__:
            if data in parameters:
                self.__dict__[data] = parameters[data]
        self.ssda_network = False  # A MODIFIER
        self.task_manager = self._init_task_manager(n_classes=self.output_size)
        self.split_name = (
            self._check_split_wording()
        )  # Used only for retro-compatibility

    def _check_split_wording(self):
        """Finds if MAPS structure uses 'fold-X' or 'split-X' folders."""

        if len(list(self.maps_path.glob("fold-*"))) > 0:
            return "fold"
        else:
            return "split"

    def _init_task_manager(self, df=None, n_classes=None):
        from clinicadl.utils.task_manager import (
            ClassificationManager,
            ReconstructionManager,
            RegressionManager,
        )

        if self.network_task == "classification":
            if n_classes is not None:
                return ClassificationManager(self.mode, n_classes=n_classes)
            else:
                return ClassificationManager(self.mode, df=df, label=self.label)
        elif self.network_task == "regression":
            return RegressionManager(self.mode)
        elif self.network_task == "reconstruction":
            return ReconstructionManager(self.mode)
        else:
            raise NotImplementedError(
                f"Task {self.network_task} is not implemented in ClinicaDL. "
                f"Please choose between classification, regression and reconstruction."
            )

    def _init_split_manager(self, split_list=None):
        from clinicadl.utils import split_manager

        split_class = getattr(split_manager, self.validation)
        args = list(
            split_class.__init__.__code__.co_varnames[
                : split_class.__init__.__code__.co_argcount
            ]
        )
        args.remove("self")
        args.remove("split_list")
        kwargs = {"split_list": split_list}
        for arg in args:
            kwargs[arg] = self.__dict__[arg]
        return split_class(**kwargs)

    def get_parameters(self):
        """Returns the training parameters dictionary."""
        json_path = self.maps_path / "maps.json"
        return read_json(json_path)

    def initiate_maps(self):
        task_manager = self._check_args()
        self.__dict__["tsv_path"] = Path(self.__dict__["tsv_path"])

        self.split_name = "split"  # Used only for retro-compatibility
        if cluster.master:
            if (
                self.maps_path.is_dir() and self.maps_path.is_file()
            ) or (  # Non-folder file
                self.maps_path.is_dir()
                and list(self.maps_path.iterdir())  # Non empty folder
            ):
                raise MAPSError(
                    f"You are trying to create a new MAPS at {self.maps_path} but "
                    f"this already corresponds to a file or a non-empty folder. \n"
                    f"Please remove it or choose another location."
                )
            self.maps_path = Path(self.maps_path)
            print(self.maps_path)
            (self.maps_path / "groups").mkdir(parents=True)

            logger.info(f"A new MAPS was created at {self.maps_path}")

            print(self.maps_path)
            self.write_parameters()
            print(self.maps_path)
            self._write_requirements_version()

            self._write_training_data()
            self._write_train_val_groups()
            self._write_information()

    def write_parameters(self, verbose=True):
        """Write JSON files of parameters."""
        json_path = Path(self.maps_path)
        logger.debug("Writing parameters...")
        json_path.mkdir(parents=True, exist_ok=True)
        self.change_path_to_str()
        json_data = json.dumps(self.__dict__, skipkeys=True, indent=4)
        json_path = json_path / "maps.json"
        if verbose:
            logger.info(f"Path of json file: {json_path}")
        with json_path.open(mode="w") as f:
            f.write(json_data)

    def _write_requirements_version(self):
        """Writes the environment.txt file."""
        logger.debug("Writing requirement version...")
        try:
            env_variables = subprocess.check_output("pip freeze", shell=True).decode(
                "utf-8"
            )
            print(self.maps_path)
            with (Path(self.maps_path) / "environment.txt").open(mode="w") as file:
                file.write(env_variables)
        except subprocess.CalledProcessError:
            logger.warning(
                "You do not have the right to execute pip freeze. Your environment will not be written"
            )

    def _write_training_data(self):
        """Writes the TSV file containing the participant and session IDs used for training."""
        logger.debug("Writing training data...")
        from clinicadl.utils.caps_dataset.data import load_data_test

        self.change_str_to_path()
        train_df = load_data_test(
            self.tsv_path,
            self.diagnoses,
            baseline=False,
            multi_cohort=self.multi_cohort,
        )
        train_df = train_df[["participant_id", "session_id"]]
        if self.transfer_path:
            transfer_train_path = self.transfer_path / "groups" / "train+validation.tsv"
            transfer_train_df = pd.read_csv(transfer_train_path, sep="\t")
            transfer_train_df = transfer_train_df[["participant_id", "session_id"]]
            train_df = pd.concat([train_df, transfer_train_df])
            train_df.drop_duplicates(inplace=True)
        train_df.to_csv(
            self.maps_path / "groups" / "train+validation.tsv", sep="\t", index=False
        )

    def _write_train_val_groups(self):
        """Defines the training and validation groups at the initialization"""
        logger.debug("Writing training and validation groups...")
        self.change_str_to_path()
        split_manager = self._init_split_manager()
        for split in split_manager.split_iterator():
            for data_group in ["train", "validation"]:
                df = split_manager[split][data_group]
                print(self.maps_path)
                group_path = (
                    Path(self.maps_path)
                    / "groups"
                    / data_group
                    / f"{self.split_name}-{split}"
                )
                group_path.mkdir(parents=True, exist_ok=True)

                columns = ["participant_id", "session_id", "cohort"]
                if self.label is not None:
                    columns.append(self.label)
                df.to_csv(group_path / "data.tsv", sep="\t", columns=columns)
                self.write_parameters(
                    verbose=False,
                )

    def _write_information(self):
        """
        Writes model architecture of the MAPS in MAPS root.
        """
        from datetime import datetime

        import clinicadl.utils.network as network_package

        model_class = getattr(network_package, self.architecture)
        args = list(
            model_class.__init__.__code__.co_varnames[
                : model_class.__init__.__code__.co_argcount
            ]
        )
        args.remove("self")
        kwargs = dict()
        for arg in args:
            kwargs[arg] = self.__dict__[arg]
        kwargs["gpu"] = False

        model = model_class(**kwargs)

        file_name = "information.log"

        with (Path(self.maps_path) / file_name).open(mode="w") as f:
            f.write(f"- Date :\t{datetime.now().strftime('%d %b %Y, %H:%M:%S')}\n\n")
            f.write(f"- Path :\t{self.maps_path}\n\n")
            # f.write("- Job ID :\t{}\n".format(os.getenv('SLURM_JOBID')))
            f.write(f"- Model :\t{model.layers}\n\n")

        del model

    def check_existing_maps2(self):
        self.json_path = Path(self.maps_path / "maps.json")
        if not self.json_path.is_file():
            raise MAPSError(
                f"MAPS was not found at {self.maps_path}."
                f"To initiate a new MAPS please give a train_dict."
            )
        test_parameters = read_json(self.json_path)
        test_parameters = change_str_to_path(test_parameters)
        for data in self.__dict__:
            if data in test_parameters:
                self.__dict__[data] = test_parameters[data]
        self.ssda_network = False  # A MODIFIER
        self.split_name = (
            self._check_split_wording()
        )  # Used only for retro-compatibility

        return self._init_task_manager(n_classes=self.output_size)

    def _check_args(self):
        """
        Check the training parameters integrity
        """
        logger.debug("Checking arguments...")
        mandatory_arguments = [
            "caps_directory",
            "tsv_path",
            "preprocessing_dict",
            "mode",
            "network_task",
        ]
        for arg in mandatory_arguments:
            if self.__dict__[arg] is None:
                raise ClinicaDLArgumentError(
                    f"The values of mandatory arguments {mandatory_arguments} should be set. "
                    f"No value was given for {arg}."
                )
        self.add_default_values()
        self.change_str_to_path()
        if self.__dict__["gpu"]:
            check_gpu()
        elif self.__dict__["amp"]:
            raise ClinicaDLArgumentError(
                "AMP is designed to work with modern GPUs. Please add the --gpu flag."
            )

        _, transformations = get_transforms(
            normalize=self.normalize,
            size_reduction=self.size_reduction,
            size_reduction_factor=self.size_reduction_factor,
        )

        split_manager = self._init_split_manager(None)
        train_df = split_manager[0]["train"]
        if "label" not in self.__dict__:
            self.__dict__["label"] = None

        task_manager = self._init_task_manager(df=train_df)

        if self.__dict__["architecture"] == "default":
            self.__dict__["architecture"] = task_manager.get_default_network()
        if "selection_threshold" not in self.__dict__:
            self.__dict__["selection_threshold"] = None
        if (
            "label_code" not in self.__dict__ or len(self.__dict__["label_code"]) == 0
        ):  # Allows to set custom label code in TOML
            self.__dict__["label_code"] = task_manager.generate_label_code(
                train_df, self.label
            )

        full_dataset = return_dataset(
            self.caps_directory,
            train_df,
            self.preprocessing_dict,
            multi_cohort=self.multi_cohort,
            label=self.label,
            label_code=self.__dict__["label_code"],
            train_transformations=None,
            all_transformations=transformations,
        )
        self.__dict__.update(
            {
                "num_networks": full_dataset.elem_per_image,
                "output_size": task_manager.output_size(
                    full_dataset.size, full_dataset.df, self.label
                ),
                "input_size": full_dataset.size,
            }
        )

        self.__dict__["seed"] = get_seed(self.__dict__["seed"])

        if self.__dict__["num_networks"] < 2 and self.multi_network:
            raise ClinicaDLConfigurationError(
                f"Invalid training configuration: cannot train a multi-network "
                f"framework with only {self.__dict__['num_networks']} element "
                f"per image."
            )
        possible_selection_metrics_set = set(task_manager.evaluation_metrics) | {"loss"}
        if not set(self.__dict__["selection_metrics"]).issubset(
            possible_selection_metrics_set
        ):
            raise ClinicaDLConfigurationError(
                f"Selection metrics {self.__dict__['selection_metrics']} "
                f"must be a subset of metrics used for evaluation "
                f"{possible_selection_metrics_set}."
            )
        return task_manager

    def change_str_to_path(self):
        """
        For all paths in the dictionnary, it changes the type from str to pathlib.Path.

        Paramaters
        ----------
        toml_dict: Dict[str, Dict[str, Any]]
            Dictionary of options as written in a TOML file, with type(path)=str

        Returns
        -------
            Updated TOML dictionary with type(path)=pathlib.Path
        """
        for key, value in self.__dict__.items():
            if type(value) == Dict:
                for key2, value2 in value.items():
                    if (
                        key2.endswith("tsv")
                        or key2.endswith("dir")
                        or key2.endswith("directory")
                        or key2.endswith("path")
                        or key2.endswith("json")
                        or key2.endswith("location")
                    ):
                        if value2 == "":
                            self.__dict__[value][key2] = False
                        else:
                            self.__dict__[value][key2] = Path(value2)
            else:
                if (
                    key.endswith("tsv")
                    or key.endswith("dir")
                    or key.endswith("directory")
                    or key.endswith("path")
                    or key.endswith("json")
                    or key.endswith("location")
                ):
                    if value == "":
                        self.__dict__[key] = False
                    elif value == None:
                        self.__dict__[key] = False
                    elif type(value) is not bool:
                        self.__dict__[key] = Path(value)

    def change_path_to_str(self):
        """
        For all paths in the dictionnary, it changes the type from pathlib.Path to str.

        Paramaters
        ----------
        toml_dict: Dict[str, Dict[str, Any]]
            Dictionary of options as written in a TOML file, with type(path)=pathlib.Path

        Returns
        -------
            Updated TOML dictionary with type(path)=str
        """
        for key, value in self.__dict__.items():
            if type(value) == Dict:
                for key2, value2 in value.items():
                    if (
                        key2.endswith("tsv")
                        or key2.endswith("dir")
                        or key2.endswith("directory")
                        or key2.endswith("path")
                        or key2.endswith("json")
                        or key2.endswith("location")
                    ):
                        if value2 == False:
                            self.__dict__[value][key2] = ""
                        elif isinstance(value2, Path):
                            self.__dict__[value][key2] = value2.as_posix()
            else:
                if (
                    key.endswith("tsv")
                    or key.endswith("dir")
                    or key.endswith("directory")
                    or key.endswith("path")
                    or key.endswith("json")
                    or key.endswith("location")
                ):
                    if value == False:
                        self.__dict__[key] = ""
                    elif isinstance(value, Path):
                        self.__dict__[key] = value.as_posix()

    def add_default_values(self):
        """
        Updates the training parameters defined by the user with the default values in missing fields.

        Args:
            user_dict: dictionary of training parameters defined by the user.

        Returns:
            dictionary of values ready to use for the training process.
        """
        import toml

        from clinicadl.utils.maps_manager.maps_manager_utils import remove_unused_tasks

        task = self.__dict__["network_task"]
        # read default values
        clinicadl_root_dir = (Path(__file__) / "../../..").resolve()
        config_path = clinicadl_root_dir / "resources" / "config" / "train_config.toml"
        config_dict = toml.load(config_path)

        # task dependent
        config_dict = remove_unused_tasks(config_dict, task)
        # Check that TOML file has the same format as the one in resources
        for section_name in config_dict:
            for key in config_dict[section_name]:
                if key not in self.__dict__:  # Add value if not present in user_dict
                    self.__dict__[key] = config_dict[section_name][key]

        # Hard-coded options
        if self.__dict__["n_splits"] and self.__dict__["n_splits"] > 1:
            self.__dict__["validation"] = "KFoldSplit"
        else:
            self.__dict__["validation"] = "SingleSplit"
