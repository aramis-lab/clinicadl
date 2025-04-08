import json
import shutil
import subprocess
from abc import abstractmethod
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
    COMPUTATIONAL,
    DATA,
    DESCRIPTION,
    ENVIRONMENT,
    GROUPS,
    INFORMATION,
    MAPS,
    METRICS,
    MODEL,
    OPTIMIZATION,
    OPTIMIZER,
    PARTICIPANT_ID,
    PREDICTIONS,
    SESSION_ID,
    SPLIT,
    TMP,
    TRAIN,
    TRAINING,
    VALIDATION,
)
from clinicadl.losses import ImplementedLoss, get_loss_function_config
from clinicadl.metrics.metrics import MetricConfig, Metrics, MonaiMetric
from clinicadl.model import ClinicaDLModel
from clinicadl.networks import ImplementedNetwork, get_network_config
from clinicadl.optim.config import OptimizationConfig
from clinicadl.optim.optimizers import ImplementedOptimizer, get_optimizer_config
from clinicadl.splitter.split import Split
from clinicadl.tsvtools.utils import df_to_tsv, tsv_to_df
from clinicadl.utils import cluster
from clinicadl.utils.computational.config import ComputationalConfig
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
    ClinicaDLDataLeakageError,
    ClinicaDLMAPSError,
)
from clinicadl.utils.json import path_encoder, update_json
from clinicadl.utils.typing import PathType

TRAIN_VAL = [TRAIN, VALIDATION]


class Directory:
    def __init__(self, path: PathType):
        self.path = Path(path)

    def exists(self):
        return self.path.is_dir()

    def is_empty(self):
        return not any(self.path.iterdir())


class BestMetricDataGroup(Directory):
    def __init__(self, name: str, parent_dir: PathType):
        self.name = name
        super().__init__(Path(parent_dir) / name)

    @property
    def description_log(self) -> Path:
        return self.path / (DESCRIPTION + LOG)

    @property
    def metrics_tsv(self) -> Path:
        return self.path / (METRICS + TSV)

    @property
    def prediction_tsv(self) -> Path:
        return self.path / (PREDICTIONS + TSV)

    @property
    def caps_output(self) -> Path:
        return self.path / "CAPSOutput"

    def create(self, datagroup: str, caps_dir: PathType, df: pd.DataFrame):
        if self.exists():
            raise ClinicaDLConfigurationError(
                f"Data group '{self.name}' already exists."
            )

        self.path.mkdir(parents=True)
        self._write_description_log(datagroup=datagroup, caps_dir=caps_dir, df=df)

    def _write_description_log(
        self, datagroup: str, caps_dir: PathType, df: pd.DataFrame
    ):
        """
        Write description log file associated to a data group.

        Args:
            log_dir (str): path to the log file directory.
            data_group (str): name of the data group used for the task.
            caps_dict (dict[str, str]): Dictionary of the CAPS folders used for the task
            df (pd.DataFrame): DataFrame of the meta-data used for the task.
        """
        if self.description_log.exists():
            raise ClinicaDLConfigurationError(f"Description log already exists.")

        with self.description_log.open(mode="w") as f:
            f.write(f"Prediction {datagroup} group - {datetime.now()}\n")
            f.write(f"Data loaded from CAPS directories: {caps_dir}\n")
            f.write(f"Number of participants: {df.participant_id.nunique()}\n")
            f.write(f"Number of sessions: {len(df)}\n")


class BestMetric(Directory):
    def __init__(self, metric: MetricConfig, parent_dir: PathType):
        self.metric = metric
        super().__init__(Path(parent_dir) / (BEST + "-" + metric.name))

        self.train = BestMetricDataGroup(TRAIN, self.path)
        self.val = BestMetricDataGroup(VALIDATION, self.path)
        self.data_groups: Dict[str, BestMetricDataGroup] = {}

    @property
    def model(self) -> Path:
        return self.path / (MODEL + PTH + TAR)

    def create(self, split: Split):
        if self.exists():
            raise ClinicaDLConfigurationError(
                f"Best metric '{self.metric.name}' already exists."
            )

        self.path.mkdir(parents=True)
        self.train.create(
            datagroup=TRAIN,
            caps_dir=split.train_dataset.directory,
            df=split.train_dataset.df,
        )
        self.val.create(
            datagroup=VALIDATION,
            caps_dir=split.val_dataset.directory,
            df=split.val_dataset.df,
        )

    def create_data_group(self, name: str):
        if name in self.data_groups:
            raise ClinicaDLConfigurationError(f"Data group '{name}' already exists.")

        tmp_group = BestMetricDataGroup(name, self.path)

        if tmp_group.exists():
            raise ClinicaDLConfigurationError(f"Data group '{name}' already exists.")

        tmp_group.path.mkdir(parents=True)
        self.data_groups[name] = tmp_group

        return tmp_group


class SplitDir(Directory):
    def __init__(self, num: int, best_metrics: list[MetricConfig], maps_path: PathType):
        self.number = num
        super().__init__(Path(maps_path) / (SPLIT + "-" + str(num)))

        self.logs = TrainingLogs(self.path)
        self.tmp = TmpDir(self.path)

        self.best_metrics: Dict[str, BestMetric] = {}
        for metric in best_metrics:
            self.best_metrics[metric.name] = BestMetric(metric, self.path)

        # TODO: add somethin to write split info

    @property
    def split_json(self) -> Path:
        return self.path / (SPLIT + JSON)

    def create(self, split: Split):
        if self.exists():
            raise ClinicaDLConfigurationError(f"Split '{self.number}' already exists.")
        self.path.mkdir(parents=True)
        self._write_split_json(split)
        for metric in self.best_metrics.values():
            metric.create(split)

    def _write_split_json(self, split: Split):
        """Writes the maps.json file."""

        dict_ = split.model_dump(exclude={"train_loader", "val_loader"})

        dict_["val_dataset"] = split.val_dataset.describe()
        dict_["train_dataset"] = split.train_dataset.describe()

        update_json(
            self.split_json, dict_=dict_
        )  # called to add data to the split.json


class TrainingLogs(Directory):
    def __init__(self, parent_dir: PathType):
        super().__init__(Path(parent_dir) / "training_logs")

    @property
    def tensorboard(self) -> Path:
        return self.path / "tensorboard"

    @property
    def training_tsv(self) -> Path:
        return self.path / (TRAINING + TSV)


class TmpDir(Directory):
    def __init__(self, parent_dir: PathType):
        super().__init__(Path(parent_dir) / TMP)

    @property
    def checkpoint(self) -> Path:
        return self.path / (CHECKPOINT + PTH + TAR)

    @property
    def optimizer(self) -> Path:
        return self.path / (OPTIMIZER + PTH + TAR)


class BaseDataGroup(Directory):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def data_tsv(self) -> Path:
        return self.path / (DATA + TSV)

    @property
    def maps_json(self) -> Path:
        return self.path / (MAPS + JSON)

    def create(self, dataset: CapsDataset):
        if self.exists():
            raise ClinicaDLConfigurationError(
                f"Data group '{self.name}' already exists."
            )

        self.path.mkdir(parents=True)
        df_to_tsv(self.data_tsv, dataset.df)
        # self._write_maps_json(dataset) TODO: check what's in the maps.json here


class DataGroup(BaseDataGroup):
    def __init__(self, name: str, parent_dir: PathType):
        self.name_ = name

        super().__init__(Path(parent_dir) / name)

    @property
    def name(self) -> str:
        return self.name_


class TrainValDataGroup(BaseDataGroup):
    def __init__(self, name: str, parent_dir: PathType, split: int):
        self.name_ = name
        self.split = split
        super().__init__(Path(parent_dir) / name / f"{SPLIT}-{split}")

    @property
    def name(self) -> str:
        return self.name_


class Maps(Directory):
    def __init__(self, maps_path: PathType):
        super().__init__(maps_path)

        self.splits: Dict[int, SplitDir] = {}
        self.data_groups: Dict[str, Union[Dict[int, BaseDataGroup], BaseDataGroup]] = {}

    @property
    def groups_dir(self) -> Path:
        return self.path / GROUPS

    @property
    def split_list(self) -> list[int]:
        return [int(x.name.split("-")[1]) for x in self.path.iterdir() if x.is_dir()]

    @property
    def group_list(self) -> list[str]:
        return [x.name for x in self.groups_dir.iterdir() if x.is_dir()]

    @property
    def train_val_tsv(self) -> Path:
        return self.path / f"{TRAIN}+{VALIDATION}{TSV}"

    @property
    def maps_json(self) -> Path:
        return self.path / (MAPS + JSON)

    @property
    def model_json(self) -> Path:
        return self.path / (MODEL + JSON)

    @property
    def computational_json(self) -> Path:
        return self.path / (COMPUTATIONAL + JSON)

    @property
    def optimization_json(self) -> Path:
        return self.path / (OPTIMIZATION + JSON)

    def create_data_group(self, name: str, dataset: CapsDataset):
        if name in self.data_groups:
            raise ClinicaDLConfigurationError(f"Data group '{name}' already exists.")

        data_group = DataGroup(name, self.groups_dir)
        data_group.create(dataset=dataset)
        self.data_groups[name] = data_group

    def create_split(self, split: Split, best_metrics: list[MetricConfig]):
        if split.index in self.splits:
            raise ClinicaDLConfigurationError(f"Split '{split.index}' already exists.")

        split_dir = SplitDir(split.index, best_metrics, self.path)
        split_dir.create(split)
        self.splits[split.index] = split_dir

        train_group = TrainValDataGroup(TRAIN, self.path, split.index)
        train_group.create(split.train_dataset)
        self.data_groups[TRAIN] = {split.index: train_group}

        val_group = TrainValDataGroup(VALIDATION, self.path, split.index)
        val_group.create(split.val_dataset)
        self.data_groups[VALIDATION] = {split.index: val_group}

    def create(self):
        if self.exists():
            raise ClinicaDLConfigurationError(
                f"Maps directory ({self.path})already exists."
            )

        self.path.mkdir(parents=True, exist_ok=True)
        self.groups_dir.mkdir(parents=True)
        self._write_requirements_version()

    def _write_requirements_version(self):
        """Writes the environment.txt file."""
        try:
            env_variables = subprocess.check_output("pip freeze", shell=True).decode(
                "utf-8"
            )
            with (self.path / "environment.txt").open(mode="w") as file:
                file.write(env_variables)
        except subprocess.CalledProcessError:
            raise ClinicaDLConfigurationError(
                "You do not have the right to execute pip freeze. Your environment will not be written"
            )

    def read_maps(self) -> dict:
        """Reads the maps.json file."""
        if not self.maps_json.is_file():
            raise ClinicaDLConfigurationError("Could not find maps.json")

        with open(self.maps_json, "r") as file:
            x = json.load(file)
            return json.loads(x)
