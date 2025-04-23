from pathlib import Path
from typing import List, Optional, Sequence, Union

from pydantic import NonNegativeFloat, field_validator

from clinicadl.data.datasets.caps_dataset import CapsDataset
from clinicadl.splitter.split import Split
from clinicadl.splitter.splitter.splitter import (
    Splitter,
    SplitterConfig,
    SubjectsSessionsSplit,
)


class SingleSplitConfig(SplitterConfig):
    """
    Configuration for single split.
    """

    _json_name: str = "single_split_config"

    n_test: NonNegativeFloat
    stratification: Union[List[str], bool]
    p_categorical_threshold: NonNegativeFloat
    p_continuous_threshold: NonNegativeFloat

    @field_validator("p_categorical_threshold", "p_continuous_threshold", mode="after")
    @classmethod
    def validate_thresholds(cls, value: Union[float, int], ctx) -> float:
        if not (0 <= value <= 1):
            raise ValueError(f"'{ctx.field_name}' must be between 0 and 1, got {value}")
        return value

    def _check_split_dirs(self) -> None:
        """Checks the split directory."""
        self._check_split_dir(self.split_dir)


class SingleSplit(Splitter):
    @property
    def _associated_config(self) -> type[SingleSplitConfig]:
        """The config class associated to the splitter."""
        return SingleSplit

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
        return self._get_split(dataset)[0]

    def _read_splits(self) -> List[SubjectsSessionsSplit]:
        """
        Load the split from the tsv files in 'split_dir'.
        """
        return [self._read_split(self.split_dir)]
