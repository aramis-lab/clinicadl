from pathlib import Path
from typing import Generator, List, Optional, Sequence, Tuple, Union

from pydantic import PositiveInt

from clinicadl.dataset.datasets.caps_dataset import CapsDataset
from clinicadl.splitter.split import Split
from clinicadl.splitter.splitter import Splitter, SplitterConfig, SubjectsSessionsSplit


class KFoldConfig(SplitterConfig):
    """
    Configuration for K-Fold cross-validation splits.
    """

    json_name: str = "kfold_config.json"
    subset_name: str = "validation"
    n_splits: PositiveInt = 5

    @property
    def pattern(self):
        return f"{self.n_splits}_fold"


class KFold(Splitter):
    """
    Handles K-Fold cross-validation with optional stratification and demographic balancing.
    Allows saving, reading, and iterating over splits for reproducibility.
    """

    json_name = "kfold_config.json"

    def __init__(self, split_dir: Path):
        """
        Initialize KFold with a dataset.

        Parameters
        ----------
        dataset : CapsDataset
            Dataset to split for cross-validation.
        """

        super().__init__(split_dir=split_dir)

    def _config(self, **args) -> KFoldConfig:
        return KFoldConfig(**args)

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
        return [
            self._read_split(self.split_dir / f"split-{i}")
            for i in range(self.config.n_splits)
        ]

    def get_splits(
        self, dataset: CapsDataset, splits: Optional[Sequence[int]] = None
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

        if not self.config or self.config.n_splits is None:
            raise ValueError(
                "No splits found, you must first run the function 'make_splits' to split your dataset, "
                "or make sure you use a working split dir ."
            )

        self.check_dataset_and_tsv_consistency(dataset)

        if splits is None:
            splits = list(range(self.config.n_splits))

        for split in splits:
            if split not in range(self.config.n_splits):
                raise ValueError(
                    f"Split-{split} doesn't exist. There are {self.config.n_splits} splits, numbered from 0 to {self.config.n_splits-1}."
                )
            yield self._get_split(split_id=split, dataset=dataset)
