from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, List, Optional, Sequence, Union

import pandas as pd
from pydantic import (
    field_validator,
)

from clinicadl.data.datasets.caps_dataset import CapsDataset
from clinicadl.data.datasets.types import Dataset
from clinicadl.dictionary.suffixes import JSON, TSV
from clinicadl.dictionary.words import BASELINE
from clinicadl.splitter.split import Split
from clinicadl.utils.config import ClinicaDLConfig


class SubjectsSessionsSplit(ClinicaDLConfig):
    """
    Dataclass to store train and validation splits for subjects and sessions.
    """

    training: pd.DataFrame
    validation: pd.DataFrame


class SplitterConfig(ClinicaDLConfig, ABC):
    """
    Base abstract config class for splitters.
    """

    _training_subset_name: str = "train"
    _json_name: str

    split_dir: Path
    subset_name: str
    stratification: Union[str, List[str], bool]
    longitudinal: bool
    seed: int

    @field_validator("split_dir", mode="after")
    @classmethod
    def validate_split_dir(cls, v: Path) -> Path:
        """Creates 'split_dir' if it doesn't exist."""
        if not v.is_dir():
            v.mkdir(parents=True, exist_ok=True)
        return v

    @classmethod
    def from_split_dir(cls, split_dir: Path) -> SplitterConfig:
        """
        Reads a split directory.
        """
        json_path = (split_dir / cls._json_name).with_suffix(JSON)

        try:
            config = cls.from_json(json_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"No configuration file found in '{split_dir}'. It was expected at {json_path}. "
                "Please rerun clinicadl.splitter.make_split or clinicadl.splitter.make_kfold "
                "to have a proper split directory."
            ) from exc

        config._check_split_dirs()

        return config

    def write_json(self) -> None:  # pylint: disable=arguments-differ
        """
        Saves the split configuration in a json file.
        """
        out_json_file = (self.split_dir / self._json_name).with_suffix(JSON)
        super().write_json(out_json_file)

    @abstractmethod
    def _check_split_dirs(self) -> None:
        """Checks all subdirectories in the current split directory."""

    def _check_split_dir(self, split_path: Path) -> None:
        """
        Checks that a split directory exists and contains the required
        tsv files.
        """
        error_msg = ""

        if not split_path.is_dir():
            error_msg = f"No such directory: {split_path}."

        else:
            required_files = [
                (split_path / f"{self._training_subset_name}_{BASELINE}").with_suffix(
                    TSV
                ),
                (split_path / f"{self.subset_name}_{BASELINE}").with_suffix(TSV),
                (split_path / f"{self._training_subset_name}").with_suffix(TSV),
            ]
            if self.longitudinal:
                required_files.append(
                    (split_path / f"{self.subset_name}").with_suffix(TSV)
                )

            for file in required_files:
                if not file.is_file():
                    error_msg = f"Required file missing: {str(file)}."
                    break

        if error_msg:
            error_msg += (
                " Please rerun clinicadl.splitter.make_split or clinicadl.splitter.make_kfold to have a proper "
                "split directory."
            )
            raise FileNotFoundError(error_msg)


class Splitter(ABC):
    """
    Base abstract class for splitters.

    Parameters
    ----------
    split_dir : Path
        The split directory, returned by :py:func:`clinicadl.splitter.make_split`
        or :py:func:`clinicadl.splitter.make_kfold`.
    """

    def __init__(self, split_dir: Path):
        split_dir = Path(split_dir)
        if not split_dir.is_dir():
            raise FileNotFoundError(f"No such directory: {str(split_dir)}")

        self.split_dir = split_dir
        self.config = self._associated_config.from_split_dir(self.split_dir)
        self.subjects_sessions_split = self._read_splits()

    @property
    @abstractmethod
    def _associated_config(self) -> type[SplitterConfig]:
        """The config class associated to the splitter."""

    @abstractmethod
    def get_splits(
        self, dataset: Dataset, splits: Optional[Sequence[int]] = None
    ) -> Union[Split, Generator[Split, None, None]]:
        """
        Split the dataset and yield splits by their indices.

        Parameters
        ----------
        splits : Sequence[int]
            Indices of the splits to retrieve.

        Yields
        ------
        Split
            A :py:class:`~clinicadl.splitter.Split` object, with the training and validation datasets for
            the requested split.

        Raises
        ------
        ValueError
            If the requested split indices are out of range or no splits are available.
        IndexError
            If one of the requested split indices is out of range.
        """

    def _get_split(
        self,
        dataset: Dataset,
        split_id: int = 0,
    ) -> Split:
        """
        Splits a dataset.
        """
        subjects_sessions = self.subjects_sessions_split[split_id]
        return Split(
            index=split_id,
            split_dir=self.split_dir,
            train_dataset=dataset.subset(subjects_sessions.training),
            val_dataset=dataset.subset(subjects_sessions.validation),
        )

    @abstractmethod
    def _read_splits(self) -> List[SubjectsSessionsSplit]:
        """
        Load all splits in 'split_dir' from the tsv files.
        """

    def _read_split(self, split_path: Path) -> SubjectsSessionsSplit:
        """
        Load a single split from the tsv files in 'split_path'.
        """
        training_df = pd.read_csv(
            (split_path / f"{self.config._training_subset_name}").with_suffix(TSV),
            sep="\t",
        )
        if self.config.longitudinal:
            validation_df = pd.read_csv(
                (split_path / f"{self.config.subset_name}").with_suffix(TSV), sep="\t"
            )
        else:
            validation_df = pd.read_csv(
                (split_path / f"{self.config.subset_name}_{BASELINE}").with_suffix(TSV),
                sep="\t",
            )

        return SubjectsSessionsSplit(
            training=training_df,
            validation=validation_df,
        )
