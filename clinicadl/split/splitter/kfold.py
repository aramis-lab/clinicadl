from pathlib import Path
from typing import Generator, List, Optional, Sequence

from pydantic import PositiveInt, field_validator

from clinicadl.data.datasets.types import Dataset
from clinicadl.dictionary.words import SPLIT
from clinicadl.split.split import Split
from clinicadl.split.splitter.splitter import (
    Splitter,
    SplitterConfig,
    SubjectsSessionsSplit,
)


class KFoldConfig(SplitterConfig):
    """
    Configuration for K-Fold cross-validation splits.
    """

    _json_name: str = "kfold_config"

    n_splits: PositiveInt
    stratification: Optional[str]

    @field_validator("n_splits", mode="after")
    @classmethod
    def n_splits_validator(cls, v: int) -> int:
        """Checks that 'n_splits' is greater than 2."""
        assert v >= 2, "'n_splits' must be at least 2."
        return v

    def get_split_subdir(self, split: int, create: bool = False) -> Path:
        """
        Returns the subdirectory of a split of a K-Fold, and creates this directory if it does not
        exist yet.

        Parameters
        ----------
        split : int
            The index of the split.
        create : bool (optional, default=False)
            Create the directory if it doesn't exist.

        Returns
        -------
        Path
            The path to the split subdirectory.
        """
        split_dir = self.split_dir / f"{SPLIT}-{split}"
        if create:
            split_dir.mkdir(parents=True, exist_ok=True)

        return split_dir

    def _check_split_dirs(self) -> None:
        """Checks all the splits directories."""
        for i in range(self.n_splits):
            self._check_split_dir(self.get_split_subdir(i))


class KFold(Splitter):
    """
    To handle a K-Fold cross-validator.

    This object will read a split directory returned by :py:func:`~clinicadl.split.make_kfold`,
    and can then be used to split any :py:class:`~clinicadl.data.datasets.CapsDataset` (or
    :py:class:`~clinicadl.data.datasets.ConcatDataset`, :py:class:`~clinicadl.data.datasets.PairedDataset`,
    :py:class:`~clinicadl.data.datasets.UnpairedDataset`) using :py:meth:`~KFold.get_splits`,
    provided that all the (participant, session) pairs in the dataset are mentioned in the split directory.

    Parameters
    ----------
    split_dir : Path
        The split directory, returned by :py:func:`~clinicadl.split.make_kfold``

    FileNotFoundError
        If ``split_dir`` does not exist or if a required file is missing in this directory.

    See Also
    --------
    - :py:class:`~clinicadl.split.SingleSplit`
    """

    @property
    def _associated_config(self) -> type[KFoldConfig]:
        """The config class associated to the splitter."""
        return KFoldConfig

    def get_splits(
        self, dataset: Dataset, splits: Optional[Sequence[int]] = None
    ) -> Generator[Split, None, None]:
        """
        Splits a dataset according to the splits found in the K-Fold directory, and
        yields the splits by their indices.

        Parameters
        ----------
        dataset : Dataset
            The dataset to split. Can be a :py:class:`~clinicadl.data.datasets.CapsDataset`, :py:class:`~clinicadl.data.datasets.ConcatDataset`,
            :py:class:`~clinicadl.data.datasets.PairedDataset`, or :py:class:`~clinicadl.data.datasets.UnpairedDataset`.
        splits : Optional[Sequence[int]], (optional, default=None)
            Indices of the splits to get. If ``None``, will return all the splits.

        Yields
        ------
        Split
            The train and validation datasets for each requested split, in a :py:class:`~clinicadl.split.Split`
            object.

        Raises
        ------
        IndexError
            If one of the requested split indices is out of range.
        """
        if splits is None:
            splits = list(range(self.config.n_splits))

        for split in splits:
            if split not in range(self.config.n_splits):
                raise IndexError(
                    f"Split '{split}' doesn't exist. There are {self.config.n_splits} splits, numbered from 0 to {self.config.n_splits - 1}."
                )
            yield self._get_split(split_id=split, dataset=dataset)

    def _read_splits(self) -> List[SubjectsSessionsSplit]:
        """
        Load all splits in 'split_dir' from the tsv files.
        """
        self.config: KFoldConfig
        return [
            self._read_split(self.config.get_split_subdir(i))
            for i in range(self.config.n_splits)
        ]
