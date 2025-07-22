from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

from clinicadl.dictionary.suffixes import LOG, PTH, TAR, TSV
from clinicadl.dictionary.words import (
    BEST,
    DESCRIPTION,
    METRICS,
    MODEL,
    PREDICTIONS,
    TRAIN,
    VALIDATION,
)
from clinicadl.split.split import Split
from clinicadl.utils.exceptions import ClinicaDLConfigurationError
from clinicadl.utils.typing import PathType

from ....maps.base import Directory


class BestMetricDataGroup(Directory):
    """
    Base class representing a directory structure.

    Attributes
    ----------
        path: Path
            The directory path.
    """

    def __init__(self, name: str, parent_dir: PathType):
        self.name = name
        super().__init__(path=Path(parent_dir) / name)

    @property
    def description_log(self) -> Path:
        return (self.path / DESCRIPTION).with_suffix(LOG)

    @property
    def metrics_tsv(self) -> Path:
        return (self.path / METRICS).with_suffix(TSV)

    @property
    def predictions_tsv(self) -> Path:
        return (self.path / PREDICTIONS).with_suffix(TSV)

    @property
    def caps_output(self) -> Path:
        return self.path / "CAPSOutput"

    def create(self, datagroup: str, caps_dir: PathType, df: pd.DataFrame) -> None:
        if self.exists():
            raise ClinicaDLConfigurationError(
                f"Data group '{self.name}' already exists."
            )

        self.path.mkdir(parents=True)
        self._write_description_log(datagroup=datagroup, caps_dir=caps_dir, df=df)

    def _write_description_log(
        self, datagroup: str, caps_dir: PathType, df: pd.DataFrame
    ) -> None:
        """
        Write description log file associated to a data group.

        Args:
            log_dir (str): path to the log file directory.
            data_group (str): name of the data group used for the task.
            caps_dict (dict[str, str]): Dictionary of the CAPS folders used for the task
            df (pd.DataFrame): DataFrame of the meta-data used for the task.
        """
        if self.description_log.exists():
            raise ClinicaDLConfigurationError("Description log already exists.")

        with self.description_log.open(mode="w") as f:
            f.write(f"Prediction {datagroup} group - {datetime.now()}\n")
            f.write(f"Data loaded from CAPS directories: {caps_dir}\n")
            f.write(f"Number of participants: {df.participant_id.nunique()}\n")
            f.write(f"Number of sessions: {len(df)}\n")


class BestMetric(Directory):
    """
    Class representing the `best-<metric>` folder.
    In this folder, you can find information linked to a network selected according to the chosen metric.
    The weights of the model are available (best_model.pth.tar) and their application to different data groups can be found

    `train` and `validation` data groups are automatically created as their predictions are computed during the training procedure.
    Other groups may exist if predictions and interpretations were computed.
    The content of the data group folders depend on the operations performed, then for more information please refer to the corresponding sections.

    Attributes
    ----------
        metric: MetricConfig
            The metric configuration.
        train: BestMetricDataGroup
            The training data group.
        val: BestMetricDataGroup
            The validation data group.
        data_groups: Dict[str, BestMetricDataGroup]
            Additional and non-mandatory data groups by name.
    """

    def __init__(self, metric: str, parent_dir: PathType):
        self.metric = metric
        super().__init__(path=Path(parent_dir) / (BEST + "-" + metric))

        self.train = BestMetricDataGroup(name=TRAIN, parent_dir=self.path)
        self.val = BestMetricDataGroup(name=VALIDATION, parent_dir=self.path)
        self.data_groups: Dict[str, BestMetricDataGroup] = {}

    @classmethod
    def load(cls, parent_dir: PathType, metric: str) -> BestMetric:
        """
        Load an existing best metric directory.

        Parameters
        ----------
            metric: str
                The metric name.
            parent_dir: PathType
                Path to the parent directory.

        Returns
        -------
            BestMetric
                An instance of the BestMetric class.
        """
        best_metric = cls(parent_dir=parent_dir, metric=metric)
        for data_group_name in best_metric.data_group_list:
            data_group = BestMetricDataGroup(
                parent_dir=best_metric.path, name=data_group_name
            )

            if not data_group.exists() or data_group.is_empty():
                raise ClinicaDLConfigurationError(
                    f"The data group at {data_group.path} doesn't exist or is empty."
                )
            best_metric.data_groups[data_group.name] = data_group

        if not best_metric.train.exists():
            raise ClinicaDLConfigurationError(
                f"The train data group at {best_metric.train.path} doesn't exist."
            )
        if not best_metric.val.exists():
            raise ClinicaDLConfigurationError(
                f"The validation data group at {best_metric.val.path} doesn't exist."
            )
        return best_metric

    @property
    def model(self) -> Path:
        return (self.path / MODEL).with_suffix(PTH + TAR)

    @property
    def data_group_list(self) -> list[str]:
        """
        Returns a list of available data groups in the best metric directory.
        """
        if not self.exists():
            raise ClinicaDLConfigurationError(f"The MAPS at {self.path} doesn't exist.")
        if self.is_empty():
            return []

        return [x.name for x in self.path.iterdir() if x.is_dir()]

    def create(self, split: Split) -> None:
        """
        Create the best metric directory along with its associated training and validation groups.

        Parameters
        ----------
            split: Split
                The data split information including datasets.

        Raises
        ------
            ClinicaDLConfigurationError: If the best metric directory already exists.
        """

        if self.exists():
            raise ClinicaDLConfigurationError(
                f"Best metric '{self.metric}' already exists."
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

    def create_data_group(self, name: str) -> BestMetricDataGroup:
        """
        Create an additional data group under the best metric directory.

        Parameters
        ----------
            name: str
                The name of the new data group.

        Returns
        -------
            BestMetricDataGroup: The newly created data group.

        Raises
        ------
            ClinicaDLConfigurationError: If a data group with the same name already exists.
        """

        if name in self.data_groups:
            raise ClinicaDLConfigurationError(f"Data group '{name}' already exists.")

        tmp_group = BestMetricDataGroup(name=name, parent_dir=self.path)

        if tmp_group.exists():
            raise ClinicaDLConfigurationError(f"Data group '{name}' already exists.")

        tmp_group.path.mkdir(parents=True)
        self.data_groups[name] = tmp_group

        return tmp_group
