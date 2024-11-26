from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple, Union

from pydantic import NonNegativeInt
from torch.utils.data import DataLoader

from clinicadl.dataset.caps_dataset import CapsDataset

from .dataloader import get_dataloader
from .subset import Subset


@dataclass
class Split:
    """Dataclass that contains all the useful info on the split."""

    index: int
    train_dataset: CapsDataset
    val_dataset: CapsDataset
    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None

    def create_train_loader(
        self,
        batch_size: int,
        sampling_weights: Optional[str] = None,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
        prefetch_factor: Optional[NonNegativeInt] = None,
        dp_degree: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        self.train_loader = get_dataloader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            sampling_weights=sampling_weights,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            dp_degree=dp_degree,
            rank=rank,
        )

    def create_val_loader(
        self,
        batch_size: int,
        sampling_weights: Optional[str] = None,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
        prefetch_factor: Optional[NonNegativeInt] = None,
        dp_degree: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        self.val_loader = get_dataloader(
            dataset=self.val_dataset,
            batch_size=batch_size,
            sampling_weights=sampling_weights,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            dp_degree=dp_degree,
            rank=rank,
        )


@dataclass
class SubjectsSessionsSplit:
    """Simple dataclass to store the splits."""

    train: Sequence[Tuple[str, str]]
    validation: Sequence[Tuple[str, str]]


class KFold:
    """
    KFold can be initialized with one or multiple datasets.
    If multiple datasets, they must have the same (subject, session) keys.
    """

    def __init__(self, *args: CapsDataset):
        self.datasets = args
        if len(args) == 0:
            raise ValueError("At least one dataset should be passed.")
        self.single_dataset = len(self.datasets) == 1
        self._check_datasets_consistency(self.datasets)

        self.subjects_sessions_split = None
        self.number_splits = None

    @classmethod
    def _check_datasets_consistency(cls, datasets: Sequence[CapsDataset]) -> None:
        """
        Checks that a list of caps datasets have the same (subject, session)s.
        """
        subjects_sessions_ref = cls._get_subjects_sessions(datasets[0])
        for i in range(1, len(datasets)):
            subjects_sessions = cls._get_subjects_sessions(datasets[i])
            diff = subjects_sessions_ref.symmetric_difference(subjects_sessions)
            if len(diff) > 0:
                raise ValueError(
                    "There is a mismatch between the (subject, session) lists in the input datasets. "
                    f"Dataset 0 and dataset {i} differ by: {diff}"
                )

    @staticmethod
    def _get_subjects_sessions(dataset: CapsDataset) -> set[Tuple[str, str]]:
        """
        Extracts list of (subject, session) for the tsv.
        """
        df = dataset.df
        return set(zip(df["participant_id"], df["session_id"]))

    def make_splits(self, n_splits: int, split_name: str, p_values, etc) -> None:
        """
        Does the job of clinicadl k_fold.
        Puts the splits in SubjectsSessionsSplit objects (one for each split).
        """
        # TODO
        ...
        self.subjects_sessions_split = (
            ...
        )  # SubjectsSessionsSplit object for each split
        self.number_splits = n_splits

    def write(self, split_dir: Path) -> None:
        """
        Writes the splits in a directory.
        Should be called after self.make_splits
        """
        # TODO

    def read(self, split_dir: Path) -> None:
        """
        Reads the splits from split_dir.
        """
        # TODO
        ...
        self.subjects_sessions_split = ...
        self.number_splits = ...

    def get_splits(self, splits: Sequence[int]) -> Iterator[Split]:
        """
        To iterate on splits. Returns a generator.
        """
        if self.number_splits is None:
            raise ValueError(
                "No splits found, you must first run the method 'make_splits' to split your dataset, "
                "or use the method 'read' to recover old splits."
            )
        return (self._get_split(split) for split in splits)

    def _get_split(self, split: int) -> Union[Split, Tuple[Split, ...]]:
        """
        To get a single split.
        If multiple datasets, returns the Split objects in a tuple (following
        the input order of the datasets).
        """
        self._check_split(split)
        if self.single_dataset:
            return self._get_single_dataset_split(split, dataset_id=0)
        else:
            return tuple(
                [
                    self._get_single_dataset_split(split, dataset_id=idx)
                    for idx in range(len(self.datasets))
                ]
            )

    def _check_split(self, split: int) -> None:
        """checks if split exists."""
        if split not in range(self.number_splits):
            raise ValueError(
                f"Split-{split} doesn't exist. There are {self.number_splits} splits, numbered from 0 to {self.number_splits-1}."
            )

    def _get_single_dataset_split(
        self,
        split_id: int,
        dataset_id: int,
    ) -> Split:
        """
        To split a single dataset.
        """
        dataset = self.datasets[dataset_id]
        subjects_sessions: SubjectsSessionsSplit = self.subjects_sessions_split[
            split_id
        ]
        train_dataset = Subset(dataset, subjects_sessions=subjects_sessions.train)
        val_dataset = Subset(dataset, subjects_sessions=subjects_sessions.validation)

        split = Split(
            index=split_id,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
        return split
