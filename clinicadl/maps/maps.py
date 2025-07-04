from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from clinicadl.callbacks.factory.base import Callback
from clinicadl.callbacks.handler import CallbacksHandler
from clinicadl.data.datasets import CapsDataset
from clinicadl.dictionary.suffixes import JSON, LOG, PTH, TAR, TSV, TXT
from clinicadl.dictionary.words import (
    ARCHITECTURE,
    BEST,
    CALLBACKS,
    CAPS,
    CHECKPOINTS,
    COMPUTATIONAL,
    DATA,
    DATASET,
    ENVIRONMENT,
    EPOCH,
    GROUPS,
    LOGS,
    METRICS,
    MODEL,
    OPTIMIZATION,
    OPTIMIZER,
    OUTPUT,
    PREDICTIONS,
    SPLIT,
    SUMMARY,
    TEST,
    TMP,
    TRAIN,
    TRAINING,
    VALIDATION,
)
from clinicadl.metrics.config import MetricConfig
from clinicadl.metrics.handler import MetricsHandler
from clinicadl.model import ClinicaDLModel
from clinicadl.optim.config import OptimizationConfig
from clinicadl.split.split import Split
from clinicadl.tsvtools.utils import remove_non_empty_dir
from clinicadl.utils.computational.config import ComputationalConfig
from clinicadl.utils.exceptions import ClinicaDLConfigurationError
from clinicadl.utils.typing import PathType

from .base import Directory


class BestMetric(Directory):
    def __init__(self, parent_dir: PathType, metric: str):
        super().__init__(path=Path(parent_dir) / (BEST + "-" + metric))

    @property
    def caps_output(self) -> Path:
        return self.path / (CAPS + OUTPUT)

    @property
    def metrics_tsv(self) -> Path:
        return self.path / (METRICS + TSV)


class PredSplitDir(Directory):
    def __init__(self, num: int, parent_path: PathType):
        super().__init__(path=Path(parent_path) / (SPLIT + "-" + str(num)))

        self.best_metrics: Dict[str, BestMetric] = {}

    def create(self, metric: str):
        super().create()
        best_metric = BestMetric(parent_dir=self.path, metric=metric)
        best_metric.create()
        self.best_metrics[metric] = best_metric

    @property
    def metric_list(self) -> list[str]:
        if self.is_empty():
            return []
        return [
            x.name.split("-")[1]
            for x in self.path.iterdir()
            if x.is_dir() and x.name.startswith(BEST)
        ]

    def load(self):
        super().load()
        for metric in self.metric_list:
            best_metric = BestMetric(parent_dir=self.path, metric=metric)
            best_metric.load()
            self.best_metrics[metric] = best_metric

    @property
    def computational_json(self) -> Path:
        return self.path / (COMPUTATIONAL + JSON)


class GroupDir(Directory):
    def __init__(self, parents_path: PathType, group_name: str):
        super().__init__(path=Path(parents_path) / (TEST + group_name))

        self.splits: Dict[int, PredSplitDir] = {}

    def create(self, split: int, metric: str, dataset: CapsDataset):
        super().create()

        dataset.df.to_csv(self.data_tsv, sep="\t", index=False)

        split_dir = PredSplitDir(num=split, parent_path=self.path)
        split_dir.create(metric=metric)
        self.splits[split] = split_dir

    def load(self):
        super().load()
        for idx in self.split_list:
            split = PredSplitDir(num=idx, parent_path=self.path)
            split.load()
            self.splits[idx] = split

    @property
    def split_list(self) -> list[int]:
        if self.is_empty():
            return []
        return [
            int(x.name.split("-")[1])
            for x in self.path.iterdir()
            if x.is_dir() and x.name.startswith(SPLIT)
        ]

    @property
    def caps_dataset_json(self) -> Path:
        return self.path / (CAPS + "_" + DATASET + JSON)

    @property
    def data_tsv(self) -> Path:
        return self.path / (DATA + TSV)

    @property
    def metrics_json(self) -> Path:
        return self.path / (METRICS + JSON)


class PredictionsDir(Directory):
    def __init__(self, parents_path: PathType):
        super().__init__(path=Path(parents_path) / PREDICTIONS)

        self.groups: Dict[str, GroupDir] = {}

    def load(self):
        super().load()
        for name in self.group_list:
            group = GroupDir(parents_path=self.path, group_name=name)
            group.load()
            self.groups[name] = group

    def create_group(
        self, group_name: str, split: int, metric: str, dataset: CapsDataset
    ):
        super().create(_exists_ok=True)
        group = GroupDir(parents_path=self.path, group_name=group_name)
        group.create(split=split, metric=metric, dataset=dataset)
        self.groups[group_name] = group

    @property
    def group_list(self) -> list[str]:
        if self.is_empty():
            return []
        return [
            x.name.split("-")[1]
            for x in self.path.iterdir()
            if x.is_dir() and x.name.startswith(TEST)
        ]


class DataSplitDir(Directory):
    def __init__(self, parents_path: PathType):
        super().__init__(path=Path(parents_path) / SPLIT)

        self.df = None

    def create(self, dataset: CapsDataset):
        super().create()
        self.df = dataset.df
        self.df.to_csv(self.data_tsv, sep="\t", index=False)

    def load(self):
        super().load()
        self.df = pd.read_csv(self.data_tsv, sep="\t")
        # TODO : Add check for column and index ?

    @property
    def data_tsv(self) -> Path:
        return self.path / (DATA + TSV)


class DataTrainDir(Directory):
    def __init__(self, parents_path: PathType):
        super().__init__(path=Path(parents_path) / TRAIN)

        self.splits: Dict[int, DataSplitDir] = {}

    def create(self, dataset: CapsDataset, split: int):
        super().create(_exists_ok=True)
        split_dir = DataSplitDir(parents_path=self.path / str(split))
        split_dir.create(dataset=dataset)
        self.splits[split] = split_dir

    def load(self):
        super().load()

        for idx in range(1, 6):
            split = DataSplitDir(parents_path=self.path / str(idx))
            split.load()
            self.splits[idx] = split


class DataValDir(Directory):
    def __init__(self, parents_path: PathType):
        super().__init__(path=Path(parents_path) / VALIDATION)

        self.splits: Dict[int, DataSplitDir] = {}

    def create(self, dataset: CapsDataset, split: int):
        super().create(_exists_ok=True)
        split_dir = DataSplitDir(parents_path=self.path / str(split))
        split_dir.create(dataset=dataset)
        self.splits[split] = split_dir

    def load(self):
        super().load()

        for idx in range(1, 6):
            split = DataSplitDir(parents_path=self.path / str(idx))
            split.load()
            self.splits[idx] = split


class DataDir(Directory):
    def __init__(self, parents_path: PathType):
        super().__init__(path=Path(parents_path) / DATA)

        self.train = DataTrainDir(parents_path=self.path)
        self.val = DataValDir(parents_path=self.path)

        self.df = None

    def create(self, split: Split):
        super().create(_exists_ok=True)
        self.train.create(dataset=split.train_dataset, split=split.index)
        self.val.create(dataset=split.val_dataset, split=split.index)

    def load(self):
        super().load()
        self.train.load()
        self.val.load()
        self.df = pd.read_csv(self.data_tsv, sep="\t")

    def get_caps_dataset(self):
        return CapsDataset.from_json(self.caps_dataset_json)

    @property
    def caps_dataset_json(self) -> Path:
        return self.path / (CAPS + "_" + DATASET + JSON)

    @property
    def data_tsv(self) -> Path:
        return self.path / (DATA + TSV)


class LogsDir(Directory):
    def __init__(self, parents_path: PathType):
        super().__init__(path=Path(parents_path) / LOGS)

    @property
    def training_tsv(self) -> Path:
        return self.path / (TRAINING + TSV)

    @property
    def tensorboard(self) -> Path:
        return self.path / TENSORBOARD


class EpochDir(Directory):
    def __init__(self, parents_path: PathType, epoch: int):
        super().__init__(path=Path(parents_path) / f"{EPOCH}-{epoch}")

    def load(self):
        super().load()
        # TODO : add check ?

    @property
    def model(self) -> Path:
        return self.path / (MODEL + PTH + TAR)

    @property
    def optimizer(self) -> Path:
        return self.path / (OPTIMIZER + PTH + TAR)


class CheckpointsDir(Directory):
    def __init__(self, parents_path: PathType):
        super().__init__(path=Path(parents_path) / CHECKPOINTS)

        self.epochs: Dict[int, EpochDir] = {}

    def create_epoch(self, epoch: int):
        epoch_dir = EpochDir(parents_path=self.path, epoch=epoch)
        epoch_dir.create()
        self.epochs[epoch] = epoch_dir

    def load(self):
        super().load()

        for epoch in self.epoch_list:
            epoch_dir = EpochDir(parents_path=self.path, epoch=epoch)
            epoch_dir.load()
            self.epochs[epoch] = epoch_dir

    @property
    def epoch_list(self):
        if self.is_empty():
            return []
        return [
            int(x.name.split("-")[1])
            for x in self.path.iterdir()
            if x.is_dir() and x.name.startswith(EPOCH)
        ]


class TrainBestMetric(Directory):
    def __init__(self, parent_dir: PathType, metric: str):
        super().__init__(path=Path(parent_dir) / (BEST + "-" + metric))

    def load(self):
        super().load()
        # TODO: some check ?

    @property
    def model(self) -> Path:
        return self.path / (MODEL + PTH + TAR)

    @property
    def optimizer(self) -> Path:
        return self.path / (OPTIMIZER + PTH + TAR)

    @property
    def validation_metrics_tsv(self) -> Path:
        return self.path / (VALIDATION + "_" + METRICS + TSV)


class TmpDir(Directory):
    def __init__(self, parent_dir: PathType):
        super().__init__(path=Path(parent_dir) / TMP)
        pass

    @property
    def model(self) -> Path:
        return (self.path / MODEL).with_suffix(PTH + TAR)

    @property
    def optimizer(self) -> Path:
        return (self.path / OPTIMIZER).with_suffix(PTH + TAR)

    def remove(self) -> None:
        """Removes the temporary files."""
        if self.model.is_file():
            self.model.unlink()
        if self.optimizer.is_file():
            self.optimizer.unlink()
        if self.path.is_dir():
            self.path.rmdir()


class TrainSplitDir(Directory):
    def __init__(self, num: int, parents_path: PathType):
        super().__init__(path=Path(parents_path) / (SPLIT + "-" + str(num)))

        self.best_metrics: Dict[str, TrainBestMetric] = {}

        self.checkpoints = CheckpointsDir(parents_path=self.path)
        self.logs = LogsDir(parents_path=self.path)

        self.tmp = TmpDir(parent_dir=self.path)

    def load(self):
        super().load()
        self.checkpoints.load()
        self.logs.load()
        self.tmp.load()

        for metric in self.best_metrics_list:
            best_metric = TrainBestMetric(parent_dir=self.path, metric=metric)
            best_metric.load()
            self.best_metrics[metric] = best_metric

    def get_computational_config(self) -> ComputationalConfig:
        return ComputationalConfig.from_json(self.computational_json)

    @property
    def best_metrics_list(self):
        if self.is_empty():
            return []
        return [
            x.name.split("-")[1]
            for x in self.path.iterdir()
            if x.is_dir() and x.name.startswith("best")
        ]

    @property
    def computational_json(self) -> Path:
        return self.path / (COMPUTATIONAL + JSON)

    @property
    def summary_log(self) -> Path:
        return self.path / (SUMMARY + LOG)

    @property
    def validation_metrics_tsv(self) -> Path:
        return self.path / (VALIDATION + "_" + METRICS + TSV)


class TrainingDir(Directory):
    def __init__(self, parents_path: PathType):
        super().__init__(path=Path(parents_path) / TRAINING)

        self.data = DataDir(parents_path=self.path)
        self.splits: Dict[int, TrainSplitDir] = {}

    def create_split(self, num: int):
        split = TrainSplitDir(num=num, parents_path=self.path)
        split.create()
        self.splits[num] = split

    @property
    def split_list(self) -> list[int]:
        if self.is_empty():
            return []
        return [
            int(x.name.split("-")[1])
            for x in self.path.iterdir()
            if x.is_dir() and x.name.startswith("split")
        ]

    def load(self):
        super().load()
        self.data.load()

        for idx in self.split_list:
            split = TrainSplitDir(num=idx, parents_path=self.path)
            split.load()
            self.splits[idx] = split

    def get_callbacks(self) -> list[Callback]:
        return CallbacksHandler.from_json(self.callbacks_json)

    def get_metrics(self) -> Dict[str, MetricConfig]:
        return MetricsHandler.from_json(self.metrics_json)

    def get_computational_config(self) -> ComputationalConfig:
        return ComputationalConfig.from_json(self.computational_json)

    def get_optimization_config(self) -> OptimizationConfig:
        return OptimizationConfig.from_json(self.optimization_json)

    @property
    def computational_json(self) -> Path:
        return self.path / (COMPUTATIONAL + JSON)

    @property
    def optimization_json(self) -> Path:
        return self.path / (OPTIMIZATION + JSON)

    @property
    def callbacks_json(self) -> Path:
        return self.path / (CALLBACKS + JSON)

    @property
    def metrics_json(self) -> Path:
        return self.path / (METRICS + JSON)


class Maps(Directory):
    def __init__(self, maps_path: PathType):
        super().__init__(path=maps_path)

        # self._overwrite = overwrite
        self.predictions = PredictionsDir(parents_path=self.path)
        self.training = TrainingDir(parents_path=self.path)

    def create(self, split: Optional[Split] = None, overwrite: bool = False):
        super().create(overwrite=overwrite)
        self.predictions.create(overwrite=overwrite)
        self.training.create(overwrite=overwrite)

        if split:
            self.training.create_split(num=split.index)
            self.training.data.create(split=split)

        self._write_evironment_txt()

    def load(self):
        super().load()

        self.predictions.load()
        self.training.load()

    def get_model(self):
        return ClinicaDLModel.from_json(self.model_json)

    @property
    def architecture_log(self) -> Path:
        return self.path / (ARCHITECTURE + LOG)

    @property
    def environment_txt(self) -> Path:
        return self.path / (ENVIRONMENT + TXT)

    @property
    def model_json(self) -> Path:
        return self.path / (MODEL + JSON)

    @property
    def summary_log(self) -> Path:
        return self.path / (SUMMARY + LOG)

    def _write_evironment_txt(self) -> None:
        """Writes the installed Python packages (via `pip freeze`) to `environment.txt`."""
        try:
            env_variables = subprocess.check_output("pip freeze", shell=True).decode(
                "utf-8"
            )
            with (self.environment_txt).open(mode="w") as file:
                file.write(env_variables)
        except subprocess.CalledProcessError:
            with (self.environment_txt).open(mode="w") as file:
                file.write("pip freeze")
