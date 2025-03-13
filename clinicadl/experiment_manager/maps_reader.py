import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import pandas as pd
import torch

from clinicadl.data.datasets import CapsDataset
from clinicadl.dictionary.suffixes import JSON, LOG, PTH, TAR, TSV, TXT
from clinicadl.dictionary.words import (
    BEST,
    CHECKPOINT,
    DATA,
    DESCRIPTION,
    ENVIRONMENT,
    GROUPS,
    INFORMATION,
    MAPS,
    MODEL,
    OPTIMIZER,
    PARTICIPANT_ID,
    SESSION_ID,
    SPLIT,
    TMP,
    TRAINING,
)
from clinicadl.experiment_manager.data_group import DataGroup
from clinicadl.losses import ImplementedLoss, get_loss_function_config
from clinicadl.metrics.metrics import Metrics
from clinicadl.model import ClinicaDLModel
from clinicadl.networks import ImplementedNetwork, get_network_config
from clinicadl.optim.config import OptimizationConfig
from clinicadl.optim.optimizers import ImplementedOptimizer, get_optimizer_config
from clinicadl.splitter.split import Split
from clinicadl.tsvtools.utils import df_to_tsv, tsv_to_df
from clinicadl.utils import cluster
from clinicadl.utils.computational.computational import ComputationalConfig
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
    ClinicaDLDataLeakageError,
)
from clinicadl.utils.iotools.utils import path_encoder
from clinicadl.utils.typing import PathType


class MapsReader:
    def __init__(self, maps_path: PathType) -> None:
        self.maps_path = Path(maps_path)

    def is_maps(self):
        if not self.maps_path.is_dir():
            raise ClinicaDLArgumentError(
                f"{self.maps_path} is not a directory, this maps doesn't exists"
            )

        if not self.groups_path().is_dir():
            raise ClinicaDLArgumentError(f"No groups found in {self.groups_path()}")

        if not self.maps_json_path().is_file():
            raise ClinicaDLConfigurationError(f"No maps.json found in {self.maps_path}")

        if len(self.split_list()) == 0:
            raise ClinicaDLArgumentError("No split found in the maps folder")

        return True

    def split_list(self):
        return list(self.maps_path.glob(SPLIT + "-*"))

    def _create_maps(self, overwrite: bool = False):
        """TO COMPLETE"""
        if self.maps_path.is_dir() and len(list(self.maps_path.glob("*"))) != 0:
            if overwrite:
                shutil.rmtree(self.maps_path)
            else:
                raise ClinicaDLConfigurationError(
                    "You are trying to create a new maps folder but it already exists."
                )

        self.maps_path.mkdir(parents=True, exist_ok=True)
        self._write_requirements_version()
        self._write_json(self.maps_json_path())

    def init_split(self, split: Split, metrics: Metrics):
        """Initializes the split."""
        self._write_train_val_groups(split)
        self.write_metrics(split, metrics)  # more for the metrics chosen as ref
        self.write_training_tsv(split, metrics)

    ###### GETTER ########

    def read_maps_json(self) -> dict:
        """Reads the maps.json file."""
        if not self.maps_json_path().is_file():
            raise ClinicaDLConfigurationError("Could not find maps.json")

        with open(self.maps_json_path(), "r") as file:
            x = json.load(file)
            return json.loads(x)

    def get_config(
        self, config: Union[Type[ClinicaDLConfig], list[Type[ClinicaDLConfig]]]
    ) -> Union[ClinicaDLConfig, list[ClinicaDLConfig]]:
        """Reads the configuration file."""
        dict_ = self.read_maps_json()

        if isinstance(config, type(ClinicaDLConfig)):
            return config(**dict_)

        if isinstance(config, list):
            return [conf(**dict_) for conf in config]
        # need to know which network this is

        raise ClinicaDLConfigurationError("Invalid config type")

    def get_data_group(self, name: str, split: Optional[int] = None) -> DataGroup:
        """creates a new data_group."""
        data_group = DataGroup(name=name, split=split, maps_path=self.maps_path)
        if data_group.exists():
            return data_group

        raise ClinicaDLConfigurationError(
            f"Could not find data group {data_group.name}"
        )

    def get_train_val_df(self):
        """Loads the train and validation data groups."""
        path = self.maps_path / GROUPS / "train+validation.tsv"
        return tsv_to_df(path)

    def get_best_metric_value(
        self, split: int, metric: str = "loss", group: str = "validation"
    ) -> float:
        df = tsv_to_df(self.metrics_tsv_path(split, metric, group))
        return df.at["mean", "metric"]

    def get_model(self, split: int, selection_metric: str = "loss") -> ClinicaDLModel:
        self.model_path(split, selection_metric)
        return ClinicaDLModel()  # type: ignore

    def load_metrics(self) -> Metrics:
        return Metrics()  # type: ignore

    def get_model_info(self, dict_: dict) -> ClinicaDLModel:
        """Loads the model info from the maps.json file."""

        # NETWORK
        matching_values = [
            network for network in ImplementedNetwork if network.value in dict_.keys()
        ]
        if len(matching_values) != 1:
            raise ClinicaDLConfigurationError(
                "No matching implemented network in maps.json, please give a model to initiate your trainer"
            )
        else:
            network_config = get_network_config(
                matching_values[0], **dict_[f"{matching_values[0]}"]
            )

        # OPTIM
        matching_values = [
            optimizer
            for optimizer in ImplementedOptimizer
            if optimizer.value in dict_.keys()
        ]
        if len(matching_values) != 1:
            raise ClinicaDLConfigurationError(
                "No matching implemented optimizer in maps.json, please give a model to initiate your trainer"
            )
        else:
            optimizer_config = get_optimizer_config(
                matching_values[0], **dict_[f"{matching_values[0]}"]
            )

        # LOSS
        matching_values = [
            loss for loss in ImplementedLoss if loss.value in dict_.keys()
        ]
        if len(matching_values) != 1:
            raise ClinicaDLConfigurationError(
                "No matching implemented loss in maps.json, please give a model to initiate your trainer"
            )
        else:
            loss_config = get_loss_function_config(
                matching_values[0], **dict_[f"{matching_values[0]}"]
            )

        model = ClinicaDLModel.from_config(
            network_config=network_config,
            loss_config=loss_config,
            optimizer_config=optimizer_config,
        )

        return model

    def get_metrics_info(self, dict_: dict) -> Metrics:
        """Loads the metrics info from the maps.json file."""

        if "metrics" not in dict_.keys():
            raise ClinicaDLConfigurationError(
                "No metrics in maps.json, please give metrics to initiate your trainer"
            )

        return Metrics(**dict_["metrics"])

    def get_config_info(
        self, dict_: dict
    ) -> tuple[OptimizationConfig, ComputationalConfig]:
        if OptimizationConfig.__name__ not in dict_.keys():
            # TODO: ADD logger WARNING
            optim = OptimizationConfig()
        else:
            optim = OptimizationConfig(**dict_[OptimizationConfig.__name__])

        if ComputationalConfig.__name__ not in dict_.keys():
            # TODO: ADD logger WARNING
            comp = ComputationalConfig()
        else:
            comp = ComputationalConfig(**dict_[ComputationalConfig.__name__])

        return optim, comp

    ##### WRITERS #######

    def write_model_info(self, model: ClinicaDLModel):
        if model._network_config:
            self._update_json(self.maps_json_path(), model._network_config)

        if model._loss_config:
            self._update_json(self.maps_json_path(), model._loss_config)

        if model._optimizer_config:
            self._update_json(self.maps_json_path(), model._optimizer_config)

    def write_config_info(
        self, comp_config: ComputationalConfig, optim_config: OptimizationConfig
    ):
        self._update_json(self.maps_json_path(), comp_config)
        self._update_json(self.maps_json_path(), optim_config)

    def write_metrics_info(self, metrics: Metrics):
        """Updates the metrics in the maps.json file."""
        self._update_json(self.maps_json_path(), dict_=metrics.model_dump())

    def write_training_tsv(self, split: Split, metrics: Metrics):
        """Creates a training.tsv file."""

        self.training_logs_dir_path(split.index).mkdir(parents=True, exist_ok=True)
        metrics.training_loss.to_csv(
            self.training_tsv_path(split.index), sep="\t", index=True
        )

    def write_training_logs(self, split: Split):
        """Writes training logs to the logs directory."""

        pass

    def write_metrics(self, split: Split, metrics: Metrics):
        for metric in metrics.val.selection_metrics:
            metric = metric.value
            self.best_metric_path(split.index, str(metric)).mkdir(parents=True)

            self.metrics_data_group_path(split.index, str(metric), "train").mkdir(
                parents=True
            )

            self.metrics_data_group_path(split.index, str(metric), "validation").mkdir(
                parents=True
            )

    def _write_data_group(
        self,
        dataset: CapsDataset,
        data_group: str,
    ):
        """
        Check that a data_group is not already written and writes the characteristics of the data group
        (TSV file with a list of participant / session + JSON file containing the CAPS and the preprocessing).

        Args:
            data_group (str): name whose presence is checked.
            df (pd.DataFrame): DataFrame containing the participant_id and session_id (and label if use_labels is True)
            caps_directory (str): caps_directory if different from the training caps_directory,
            multi_cohort (bool): multi_cohort used if different from the training multi_cohort.
        """
        new_data_group = DataGroup(maps_path=self.maps_path, name=data_group)
        new_data_group.create(dataset)

    def _write_train_val_groups(self, split: Split):
        """Defines the training and validation groups at the initialization"""

        train_data_group = DataGroup(
            maps_path=self.maps_path, name="train", split=split.index
        )
        train_data_group.create(split.train_dataset)

        val_data_group = DataGroup(
            maps_path=self.maps_path, name="validation", split=split.index
        )
        val_data_group.create(split.val_dataset)

        train_val_tsv = self.maps_path / GROUPS / ("train+validation" + TSV)

        concat_ = pd.concat([train_data_group.df, val_data_group.df])

        if not train_val_tsv.exists():
            concat_[[PARTICIPANT_ID, SESSION_ID]].to_csv(
                train_val_tsv, sep="\t", index=False
            )
        else:
            existing_df = tsv_to_df(train_val_tsv)[[PARTICIPANT_ID, SESSION_ID]]
            if (
                not existing_df.sort_values(by=[PARTICIPANT_ID, PARTICIPANT_ID])
                .reset_index(drop=True)
                .equals(
                    concat_[[PARTICIPANT_ID, SESSION_ID]]
                    .sort_values(by=[PARTICIPANT_ID, PARTICIPANT_ID])
                    .reset_index(drop=True)
                )
            ):
                print(
                    existing_df.sort_values(
                        by=[PARTICIPANT_ID, PARTICIPANT_ID]
                    ).reset_index(drop=True)
                )
                print(
                    concat_[[PARTICIPANT_ID, SESSION_ID]]
                    .sort_values(by=[PARTICIPANT_ID, PARTICIPANT_ID])
                    .reset_index(drop=True)
                )
                raise ClinicaDLDataLeakageError(
                    "The train+validation.tsv already exists but is different from the current split."
                )
        # TODO : check if we need to check the train+validation.tsv every split ??

    def _write_information(self, model: ClinicaDLModel):
        """
        Writes model architecture of the MAPS in MAPS root.
        """
        file_name = "information.log"

        with (self.maps_path / file_name).open(mode="w") as f:
            f.write(f"- Date :\t{datetime.now().strftime('%d %b %Y, %H:%M:%S')}\n\n")
            f.write(f"- Path :\t{self.maps_path}\n\n")
            # f.write("- Job ID :\t{}\n".format(cluster.?))
            f.write(f"- Model :\t{model.network.layers}\n\n")

    @staticmethod
    def write_description_log(
        log_dir: Path,
        data_group: DataGroup,
        caps_dict,
        df,
    ):
        """
        Write description log file associated to a data group.

        Args:
            log_dir (str): path to the log file directory.
            data_group (str): name of the data group used for the task.
            caps_dict (dict[str, str]): Dictionary of the CAPS folders used for the task
            df (pd.DataFrame): DataFrame of the meta-data used for the task.
        """
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "description.log"
        with log_path.open(mode="w") as f:
            f.write(f"Prediction {data_group} group - {datetime.now()}\n")
            f.write(f"Data loaded from CAPS directories: {caps_dict}\n")
            f.write(f"Number of participants: {df.participant_id.nunique()}\n")
            f.write(f"Number of sessions: {len(df)}\n")

    def _write_network_weights(self):
        """TO COMPLETE"""
        pass

    def _write_optim_weights(self):
        """TO COMPLETE"""
        pass

    def write_tensor(self):
        """TO COMPLETE"""
        pass

    # def _write_maps_json(
    #     self, config: Optional[ClinicaDLConfig] = None, dict_: Optional[dict] = None
    # ):
    #     """Writes the maps.json file."""
    #     json_path = self.maps_json_path()
    #     if config or
    #     self._write_json(json_path, config, dict_)

    def _write_split_json(self, split: Split):
        """Writes the maps.json file."""
        json_path = self.split_json_path(split.index)
        # self._write_json(json_path)  # called to create split

        dict_ = split.model_dump(exclude={"train_loader", "val_loader"})

        dict_["val_dataset"] = split.val_dataset.describe()
        dict_["train_dataset"] = split.train_dataset.describe()

        self._write_json(json_path, dict_=dict_)  # called to add data to the split.json

    def _update_json(
        self,
        json_path: Path,
        config: Optional[ClinicaDLConfig] = None,
        dict_: Optional[dict] = None,
    ):
        if not json_path.is_file():
            raise FileNotFoundError("The maps.json file for this MAPS does not exist.")

        # Lire le contenu existent du fichier
        with json_path.open(mode="r") as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = {}

        # Fusionner les nouvelles données
        if config:
            new_data = config.model_dump()  # Assurez-vous que c'est bien un dict
            if hasattr(config, "name"):
                name = config.name  # type: ignore
            else:
                name = config.__class__.__name__

            existing_data.update({name: new_data})

        if dict_:
            existing_data.update(dict_)

        # Écrire les données mises à jour dans le fichier
        with json_path.open(mode="w") as file:
            json.dump(existing_data, file, indent=4, default=path_encoder)

    def _write_json(
        self,
        json_path: Path,
        config: Optional[ClinicaDLConfig] = None,
        dict_: Optional[dict] = None,
    ):
        if json_path.is_file():
            raise ClinicaDLConfigurationError(
                f"The json file {json_path} already exists"
            )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with (json_path).open(mode="w") as file:
            json.dump(
                {"maps_path": self.maps_path}, file, indent=4, default=path_encoder
            )

        self._update_json(json_path, config, dict_)

    def _write_requirements_version(self):
        """Writes the environment.txt file."""
        try:
            env_variables = subprocess.check_output("pip freeze", shell=True).decode(
                "utf-8"
            )
            with (self.maps_path / "environment.txt").open(mode="w") as file:
                file.write(env_variables)
        except subprocess.CalledProcessError:
            raise ClinicaDLConfigurationError(
                "You do not have the right to execute pip freeze. Your environment will not be written"
            )

    def _write_weights(
        self,
        state: Dict[str, Any],
        split: int,
        selection_metrics: str = "loss",
        filename: str = (CHECKPOINT + PTH + TAR),
        save_all_models: bool = False,
        epoch: int = 0,
    ):
        """
        Update checkpoint and save the best model according to a set of metrics.
        If no metrics_dict is given, only the checkpoint is saved.

        Args:
            state: state of the training (model weights, epoch...).
            metrics_dict: output of RetainBest step.
            split: split number.
            network: network number (multi-network framework).
            filename: name of the checkpoint file.
        """

        checkpoint_path = self.tmp_dir_path(split) / filename
        torch.save(state, checkpoint_path)

        if save_all_models:
            torch.save(
                state, self.all_model_dir_path(split) / f"model_epoch_{epoch}.pth.tar"
            )

        best_filename = "model.pth.tar"

        for metric in selection_metrics:
            metric_path = self.best_metric_path(split, metric)
            metric_path.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checkpoint_path, metric_path / best_filename)

        # loss_path = self.best_metric_path(split=split, metric="loss")
        # loss_path.mkdir(parents=True, exist_ok=True)
        # shutil.copyfile(checkpoint_path, loss_path / best_filename)

    def _create_data_group(
        self, name: str, caps_dataset: CapsDataset, split: Optional[int] = None
    ) -> DataGroup:
        """
        Check that a data_group is not already written and writes the characteristics of the data group
        (TSV file with a list of participant / session + JSON file containing the CAPS and the preprocessing).
        """
        data_group = DataGroup(name=name, split=split, maps_path=self.maps_path)
        if data_group.exists():
            raise ClinicaDLConfigurationError(
                f"Data group {data_group.name} already exists, please give another name to your data group"
            )

        data_group.create(caps_dataset)
        return data_group

    def save_metrics(self, split: Split, metrics: Metrics):
        """Save the metrics in the MAPS."""

        self.write_training_tsv(split, metrics)

    def print_description_log(
        self,
        split: int,
        selection_metric: str,
        data_group: str,
    ):
        """
        Print the description log associated to a prediction or interpretation.

        Args:
            data_group (str): name of the data group used for the task.
            split (int): Index of the split used for training.
            selection_metric (str): Metric used for best weights selection.
        """
        with self.description_log_path(split, selection_metric, data_group).open(
            mode="r"
        ) as f:
            content = f.read()

    def _erase_tmp(self, split):
        """Erase checkpoints of the model and optimizer at the end of training."""
        tmp_path = self.tmp_dir_path(split)
        shutil.rmtree(tmp_path)

    ##### PATH #####

    # FIRST LEVEL FILES
    def maps_json_path(self) -> Path:
        return self.maps_path / (MAPS + JSON)

    def information_log_path(self) -> Path:
        return self.maps_path / (INFORMATION + LOG)

    def environment_txt_path(self) -> Path:
        return self.maps_path / (ENVIRONMENT + TXT)

    # FIRST LEVEL DIRECTORIES
    def groups_path(self) -> Path:
        return self.maps_path / GROUPS

    def split_path(self, split: int) -> Path:
        return self.maps_path / (SPLIT + "-" + str(split))

    # SPLIT LEVEL

    def split_json_path(self, split: int) -> Path:
        return self.split_path(split) / (SPLIT + JSON)

    # SPLIT / TMP PATH
    def tmp_dir_path(self, split: int, resume: bool = False) -> Path:
        checkpoint_dir = self.split_path(split) / TMP
        if not checkpoint_dir.is_dir():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    def description_log_path(
        self, split: int, selection_metric: str, data_group: str
    ) -> Path:
        return self.metrics_data_group_path(split, selection_metric, data_group) / (
            DESCRIPTION + LOG
        )

    def optimizer_path(self, split: int, resume: bool = False) -> Path:
        return self.tmp_dir_path(split) / (OPTIMIZER + PTH + TAR)

    def checkpoint_path(self, split: int, resume: bool = False) -> Path:
        return self.tmp_dir_path(split) / (CHECKPOINT + PTH + TAR)

    # SPLIT / ALL MODELS
    def all_model_dir_path(self, split: int, resume: bool = False) -> Path:
        all_models_dir = self.split_path(split) / "all_models"
        all_models_dir.mkdir(parents=True, exist_ok=True)
        return all_models_dir

    # SPLIT / BEST METRIC DIR
    def best_metric_path(self, split: int, metric: str) -> Path:
        return self.split_path(split) / f"{BEST}-{metric}"

    def model_path(self, split: int, metric: str) -> Path:
        return self.best_metric_path(split, metric) / (MODEL + PTH + TAR)

    # SPLIT / BEST METRICS / DATA GROUP

    def metrics_data_group_path(self, split: int, metric: str, data_group: str) -> Path:
        return self.best_metric_path(split, metric) / data_group

    def prediction_tsv_path(self, split: int, metric: str, data_group: str) -> Path:
        return (
            self.metrics_data_group_path(split, metric, data_group)
            / f"{data_group}_prediction.tsv"
        )

    def caps_output_path(self, split: int, metric: str, data_group: str) -> Path:
        return self.metrics_data_group_path(split, metric, data_group) / "CapsOutput"

    def metrics_tsv_path(self, split: int, metric: str, data_group: str) -> Path:
        return (
            self.metrics_data_group_path(split, metric, data_group)
            / f"{data_group}_metrics.tsv"
        )

    def best_metric_description_log(
        self, split: int, metric: str, data_group: str
    ) -> Path:
        return self.metrics_data_group_path(split, metric, data_group) / (
            DESCRIPTION + LOG
        )

    # SPLIT / TRAINING LOGS
    def training_logs_dir_path(self, split: int) -> Path:
        return self.split_path(split) / "training_logs"

    def tensorboard_dir(self, split: int) -> Path:
        return self.training_logs_dir_path(split) / "tensorboard"

    def training_tsv_path(self, split: int) -> Path:
        return self.training_logs_dir_path(split) / (TRAINING + TSV)

    # GROUPS LEVEL

    def train_val_tsv_path(self) -> Path:
        return self.groups_path() / ("train+validation" + TSV)

    # GROUPS / DATA GROUP
    def groups_data_group_path(self, data_group: str):
        return self.groups_path() / data_group

    # GROUPS / DATA GROUP / SPLIT
    def groups_data_group_split_path(self, data_group: str, split: int) -> Path:
        return self.groups_data_group_path(data_group) / (SPLIT + "-" + str(split))

    def groups_data_group_split_tsv_path(self, data_group: str, split: int) -> Path:
        return self.groups_data_group_split_path(data_group, split) / (DATA + TSV)

    def groups_data_group_split_maps_json_path(
        self, data_group: str, split: int
    ) -> Path:
        return self.groups_data_group_split_path(data_group, split) / (MAPS + JSON)
