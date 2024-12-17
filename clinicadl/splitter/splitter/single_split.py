from pathlib import Path
from typing import List, Optional, Sequence, Union

from pydantic import PositiveInt, field_validator

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
    stratification: Union[List[str], bool] = False
    n_test: PositiveInt = 100
    p_categorical_threshold: float = 0.80
    p_continuous_threshold: float = 0.80

    @property
    def pattern(self) -> str:
        return "split"

    @field_validator("p_categorical_threshold", "p_continuous_threshold", mode="before")
    def validate_thresholds(cls, value: Union[float, int]) -> float:
        if not (0 <= value <= 1):
            raise ValueError(f"Threshold must be between 0 and 1, got {value}")
        return value


class SingleSplit(Splitter):
    def __init__(self, split_dir: Path):
        """
        Initialize Split with a dataset.

        Parameters
        ----------
        dataset : CapsDataset
            Dataset to split for cross-validation.
        """
        super().__init__(split_dir=split_dir)

    def _init_config(self, **args):
        self.config = SingleSplitConfig(**args)

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
