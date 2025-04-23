from pathlib import Path
from typing import Generator, List, Optional, Sequence, Union

from pydantic import PositiveInt

from clinicadl.data.datasets.caps_dataset import CapsDataset
from clinicadl.data.datasets.types import Dataset
from clinicadl.dictionary.words import FOLD
from clinicadl.splitter.split import Split
from clinicadl.splitter.splitter.splitter import (
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
    stratification: Union[str, bool]

    def get_fold_dir(self, fold: int) -> Path:
        """
        Returns the directory of a fold of a K-Fold, and creates the directory if it does not
        exist yet.

        Parameters
        ----------
        fold : int
            The index of the fold.

        Returns
        -------
        Path
            The path to the fold directory.
        """
        split_dir = self.split_dir / f"{FOLD}-{fold}"
        split_dir.mkdir(parents=True, exist_ok=True)
        return split_dir

    def _check_split_dirs(self) -> None:
        """Checks all the fold directories."""
        for i in range(self.n_splits):
            self._check_split_dir(self.get_fold_dir(i))


class KFold(Splitter):
    """
    Handles K-Fold cross-validation with optional stratification and demographic balancing.
    Allows saving, reading, and iterating over splits for reproducibility.
    """

    @property
    def _associated_config(self) -> type[KFoldConfig]:
        """The config class associated to the splitter."""
        return KFoldConfig

    def get_splits(
        self, dataset: Dataset, splits: Optional[Sequence[int]] = None
    ) -> Generator[Split, None, None]:
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
        if splits is None:
            splits = list(range(self.config.n_splits))

        for split in splits:
            if split not in range(self.config.n_splits):
                raise IndexError(
                    f"Split-{split} doesn't exist. There are {self.config.n_splits} splits, numbered from 0 to {self.config.n_splits - 1}."
                )
            yield self._get_split(split_id=split, dataset=dataset)

    def _read_splits(self) -> List[SubjectsSessionsSplit]:
        """
        Load all folds in 'split_dir' from the tsv files.
        """
        self.config: KFoldConfig
        return [
            self._read_split(self.config.get_fold_dir(i))
            for i in range(self.config.n_splits)
        ]
