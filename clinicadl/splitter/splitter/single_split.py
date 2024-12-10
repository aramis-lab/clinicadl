from pathlib import Path
from typing import Generator, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, NonNegativeInt, PositiveFloat

from clinicadl.dataset.datasets.caps_dataset import CapsDataset
from clinicadl.splitter.split import Split
from clinicadl.splitter.splitter.splitter import (
    Splitter,
    SplitterConfig,
    SubjectsSessionsSplit,
)


class SingleSplitConfig(SplitterConfig):
    json_name: str = "single_split_config.json"
    subset_name: str = "test"
    n_test: PositiveFloat = 100
    p_sex_threshold: float = 0.80
    p_age_threshold: float = 0.80

    @property
    def pattern(self) -> str:
        return "split"


class SingleSplit(Splitter):
    json_name = "single_split_config.json"

    def __init__(self, split_dir: Path):
        """
        Initialize Split with a dataset.

        Parameters
        ----------
        dataset : CapsDataset
            Dataset to split for cross-validation.
        """
        super().__init__(split_dir=split_dir)

    def _config(self, **args) -> SingleSplitConfig:
        return SingleSplitConfig(**args)

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
        return [self._read_split(self.split_dir)]

    def get_splits(
        self, dataset: CapsDataset, splits: Optional[Sequence[int]] = None
    ) -> Split:
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

        if not self.config:
            raise ValueError(
                "No splits found, you must first run the function 'make_splits' to split your dataset, "
                "or make sure you use a working split dir ."
            )

        self.check_dataset_and_tsv_consistency(dataset)

        return self._get_split(dataset)
