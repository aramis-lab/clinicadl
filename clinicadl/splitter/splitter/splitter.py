import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, List, Optional, Sequence, Union

import pandas as pd
from pydantic import (
    computed_field,
    field_validator,
)

from clinicadl.data.datasets.caps_dataset import CapsDataset
from clinicadl.splitter.split import Split
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.exceptions import ClinicaDLTSVError
from clinicadl.utils.json import path_encoder


class SubjectsSessionsSplit(ClinicaDLConfig):
    """
    Dataclass to store train and validation splits for subjects and sessions.
    """

    train: pd.DataFrame
    validation: pd.DataFrame

    @computed_field
    @property
    def train_val_df(self) -> pd.DataFrame:
        return pd.concat([self.train, self.validation], ignore_index=True)


class SplitterConfig(ClinicaDLConfig):
    json_name: str
    split_dir: Path
    subset_name: str
    stratification: Union[str, List[str], bool] = False
    valid_longitudinal: bool = False

    @field_validator("split_dir", mode="after")
    @classmethod
    def validate_split_dir(cls, v):
        if not isinstance(v, Path):
            v = Path(v)
        if v and not v.is_dir():
            v.mkdir(parents=True, exist_ok=True)
        return v

    def _check_split_dir(self):
        split_numero = 1
        folder_name = self.pattern
        while (self.split_dir / folder_name).is_dir():
            split_numero += 1
            folder_name = f"{self.pattern}_{split_numero}"

        self.split_dir = self.split_dir / folder_name

    @property
    @abstractmethod
    def pattern(self) -> str:
        pass

    def _write_json(self) -> None:
        """
        Save KFold configuration to JSON.
        """
        if not self.split_dir:
            raise ValueError(
                "No split directory specified, use the method 'write' to save your splits."
            )

        out_json_file = self.split_dir / self.json_name
        if out_json_file.is_file():
            raise FileExistsError(
                f"File {out_json_file} already exists, your splits may have already been written."
            )

        with out_json_file.open(mode="w") as json_file:
            json.dump(
                self.model_dump(),
                json_file,
                skipkeys=True,
                indent=4,
                default=path_encoder,
            )


class Splitter(ABC):
    def __init__(self, split_dir: Path):
        """
        Initialize Split with a dataset.

        Parameters
        ----------
        dataset : CapsDataset
            Dataset to split for cross-validation.
        """
        split_dir = Path(split_dir)

        if not split_dir.is_dir():
            raise FileNotFoundError(f"No such directory: {split_dir}")

        self.split_dir = split_dir
        self._init_config(**self._read_json())
        self.subjects_sessions_split = self._read_splits()

    @abstractmethod
    def _init_config(self, **args):
        self.config: SplitterConfig

    def _read_json(self):
        """
        Load KFold configuration from a JSON file.

        Parameters
        ----------
        split_dir : Path
            Directory containing the JSON configuration file.

        Returns
        -------
        KFoldConfig
            The configuration object loaded from the JSON file.
        """

        json_file = [json for json in self.split_dir.glob("*.json")]

        if len(json_file) > 1:
            raise ValueError(
                f"Multiple JSON files found in {self.split_dir}, please remove or rename them."
            )

        elif len(json_file) == 0:
            raise FileNotFoundError(f"No JSON file found in {self.split_dir}")

        if not json_file[0].is_file():
            raise FileNotFoundError(f"No such file: {json_file}")

        with json_file[0].open(mode="r") as file:
            dict_ = json.load(file)

            return dict_

    def _read_split(self, split_path: Path) -> SubjectsSessionsSplit:
        """
        Load a single split's train and validation sets from files.

        Parameters
        ----------
        split_dir : Path
            Directory containing split data.
        split_number : int
            The split index to load.

        Returns
        -------
        SubjectsSessionsSplit
            Object containing train and validation sets as DataFrames.
        """

        if not split_path.is_dir():
            raise FileNotFoundError(f"No such directory: {split_path}")

        try:
            train = pd.read_csv(split_path / "train_baseline.tsv", sep="\t")
            validation = pd.read_csv(
                split_path / f"{self.config.subset_name}_baseline.tsv", sep="\t"
            )  # type: ignore

        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"One or more of the required files are missing: 'train_baseline.tsv', '{self.config.subset_name}_baseline.tsv'"
            ) from exc  # type: ignore

        return SubjectsSessionsSplit(
            train=train,
            validation=validation,
        )

    @abstractmethod
    def _read_splits(self) -> List[SubjectsSessionsSplit]:
        """
        Load all splits and configuration from a directory.

        Parameters
        ----------
        split_dir : Path
            Directory containing the splits and configuration JSON file.

        Returns
        -------
        None
            Populates `subjects_sessions_split` and `config` attributes.
        """

    def check_dataset_and_tsv_consistency(self, dataset: CapsDataset):
        df1 = self.subjects_sessions_split[0].train_val_df
        df2 = dataset.df
        pairs_df1 = set(zip(df1["participant_id"], df1["session_id"]))
        pairs_df2 = set(zip(df2["participant_id"], df2["session_id"]))

        # Vérification que toutes les paires de df1 sont dans df2
        if not pairs_df1.issubset(pairs_df2):
            raise ClinicaDLTSVError(
                "Not all pairs of participants and sessions from the TSV file are present in the dataset."
                "Please check the TSV file and make sure all participants and sessions are unique."
            )

    @abstractmethod
    def get_splits(
        self, dataset: CapsDataset, splits: Optional[Sequence[int]] = None
    ) -> Union[Split, Generator[Split, None, None]]:
        """
        Yield dataset splits by their indices.

        Parameters
        ----------
        splits : Sequence[int]
            Indices of the splits to retrieve.

        Yields
        ------
        Split
            The train and validation datasets for each requested split.

        Raises
        ------
        ValueError
            If the requested split indices are out of range or no splits are available.
        """

    def _get_split(
        self,
        dataset: CapsDataset,
        split_id: int = 0,
    ) -> Split:
        """
        Retrieve a single dataset split.

        Parameters
        ----------
        split_id : int
            Index of the split to retrieve.

        Returns
        -------
        Split
            Object containing train and validation datasets for the specified split.
        """
        subjects_sessions = self.subjects_sessions_split[split_id]
        return Split(
            index=split_id,
            split_dir=self.split_dir,
            train_dataset=dataset.subset(subjects_sessions.train),
            val_dataset=dataset.subset(subjects_sessions.validation),
        )
