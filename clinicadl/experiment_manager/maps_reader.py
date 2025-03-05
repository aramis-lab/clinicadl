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
from clinicadl.metrics.metrics import Metrics
from clinicadl.model import ClinicaDLModel
from clinicadl.splitter.split import Split
from clinicadl.tsvtools.utils import tsv_to_df
from clinicadl.utils import cluster
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.exceptions import (
    ClinicaDLConfigurationError,
    ClinicaDLDataLeakageError,
)
from clinicadl.utils.iotools.utils import path_encoder
from clinicadl.utils.typing import PathType


class MapsReader:
    def __init__(self, maps_path: PathType) -> None:
        self.maps_path = Path(maps_path)

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
        self._write_maps_json()

    def init_split(self, split: Split, metrics: Metrics):
        """Initializes the split."""
        self._write_train_val_groups(split)
        self.write_metrics(split, metrics)  # more for the metrics chosen as ref
        self.write_training_tsv(split, metrics)

    ###### GETTER ########

    def read_json(self) -> dict:
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
        dict_ = self.read_json()

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

    def get_model(self, split: int, selection_metric: str = "loss") -> ClinicaDLModel:
        self.model_path(split, selection_metric)
        return ClinicaDLModel()  # type: ignore

    def load_metrics(self) -> Metrics:
        return Metrics()  # type: ignore

    ##### WRITERS #######

    def write_model_info(self, model: ClinicaDLModel):
        if model._network_config:
            self._write_maps_json(model._network_config)

        if model._loss_config:
            self._write_maps_json(model._loss_config)

        if model._optimizer_config:
            self._write_maps_json(model._optimizer_config)

    def write_split_info(self, split: Split):
        self._write_split_json(split)

    def write_training_tsv(self, split: Split, metrics: Metrics):
        """Creates a training.tsv file."""

        self.training_logs_dir_path(split.index).mkdir(parents=True, exist_ok=True)

        df_train = metrics.train.df.add_suffix("_train")
        df_valid = metrics.val.df.add_suffix("_valid")

        df_final = pd.concat([df_train, df_valid], axis=1)
        df_final.to_csv(self.training_tsv_path(split.index), sep="\t", index=True)

    def write_training_logs(self, split: Split):
        """Writes training logs to the logs directory."""

        pass

    def write_metrics(self, split: Split, metrics: Metrics):
        for metric in metrics.metrics:
            self.best_metric_path(split.index, metric.__str__()).mkdir(parents=True)
            self.metrics_data_group_path(split.index, metric.__str__(), "train").mkdir(
                parents=True
            )
            self.metrics_data_group_path(
                split.index, metric.__str__(), "validation"
            ).mkdir(parents=True)

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

    def _write_maps_json(
        self, config: Optional[ClinicaDLConfig] = None, dict_: Optional[dict] = None
    ):
        """Writes the maps.json file."""
        json_path = self.maps_json_path()
        self._write_json(json_path, config, dict_)

    def _write_split_json(self, split: Split):
        """Writes the maps.json file."""
        json_path = self.split_json_path(split.index)
        self._write_json(json_path)  # called to create split

        dict_ = split.model_dump(exclude={"train_loader", "val_loader"})

        dict_["val_dataset"] = split.val_dataset.describe()
        dict_["train_dataset"] = split.train_dataset.describe()

        self._write_json(json_path, dict_=dict_)  # called to add data to the split.json
        print(self.split_json_path(split=split.index))

    def _write_json(
        self,
        json_path: Path,
        config: Optional[ClinicaDLConfig] = None,
        dict_: Optional[dict] = None,
    ):
        if config or dict_:  # the maps is supposed to already exists to write more info
            if not json_path.is_file():
                raise ClinicaDLConfigurationError(
                    "The maps.json file for this MAPS does not exist."
                )
            with (json_path).open(mode="w") as file:
                if config:
                    json.dump(
                        config.model_dump_json(indent=4),
                        file,
                        indent=4,
                        default=path_encoder,
                    )
                if dict_:
                    json.dump(dict_, file, indent=4, default=path_encoder)

        else:  # the maps is not supposed to exists
            if json_path.is_file():
                raise ClinicaDLConfigurationError(
                    f"The json file {json_path} already exists"
                )
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with (json_path).open(mode="w") as file:
                json.dump(
                    {"maps_path": self.maps_path}, file, indent=4, default=path_encoder
                )

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
        metrics: Metrics,
        network: Optional[int] = None,
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
        if network is not None:
            best_filename = f"network-{network}_model.pth.tar"

        for metric in metrics.metrics:
            metric_path = self.split_path(split) / f"best-{metric.__str__()}"
            metric_path.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checkpoint_path, metric_path / best_filename)

        loss_path = self.best_metric_path(split=split, metric="loss")
        loss_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkpoint_path, loss_path / best_filename)

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
