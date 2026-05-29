from __future__ import annotations

from bisect import bisect_right
from logging import getLogger
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from pydantic import field_validator
from typing_extensions import Self

from clinicadl.utils.dictionary.words import DATASET_ID
from clinicadl.utils.typing import DataFrameType

from ..structures import Sample
from .base import Dataset
from .collection import CollectionDataset, CollectionDatasetConfig
from .utils import CheckableDataset, OneSampleDataset

logger = getLogger(__name__)


class ConcatDatasetConfig(CollectionDatasetConfig):
    """
    Config class for ``ConcatDataset``.
    """

    @field_validator("datasets", mode="after")
    @classmethod
    def _check_dataset_id_column(
        cls, datasets: tuple[Dataset, ...]
    ) -> tuple[Dataset, ...]:
        return super()._check_dataset_id_column(datasets)

    @classmethod
    def _get_class(cls) -> type[ConcatDataset]:
        """Returns the class associated to this config class."""
        return ConcatDataset


class ConcatDataset(OneSampleDataset, CollectionDataset, CheckableDataset):
    """
    For assembling multiple :py:class:`~clinicadl.data.datasets.Dataset`
    (e.g., images coming from different BIDS datasets).

    ``ConcatDataset`` concatenates the input datasets, so the length of the new dataset will be equal to the sum
    of the lengths of each individual dataset.

    Parameters
    ----------
    datasets : Iterable[Dataset]
        The ``Datasets`` to concatenate.

    Examples
    --------
    .. code-block:: text

        bids_1
        ├── sub-001
        │   ├── ses-M000
        │   │   └── pet
        │   │       └── sub-001_ses-M000_pet.nii.gz
        │   ...
        ...

        bids_2
        ├── sub-A
        │   ├── ses-M003
        │   │   └── pet
        │   │       └── sub-A_ses-M003_pet.nii.gz
        │   ...
        ...

    .. code-block:: python

        from clinicadl.data.datasets import BidsDataset, ConcatDataset
        from clinicadl.io.bids import BidsFileType

        bids_1 = BidsDataset("bids_1", file_type=BidsFileType(data_type="pet", suffix="pet"))
        bids_2 = BidsDataset("bids_2", file_type=BidsFileType(data_type="pet", suffix="pet"))

        full_dataset = ConcatDataset([bids_1, bids_2])

    .. code-block:: python

        >>> len(bids_1)
        4
        >>> len(bids_2)
        8
        >>> len(full_dataset)
        12
        >>> full_dataset[0].participant, full_dataset[0].session
        ('sub-001', 'ses-M000')
        >>> full_dataset[4].participant, full_dataset[4].session
        ('sub-A', 'ses-M003')
    """

    _config_type = ConcatDatasetConfig

    def __init__(
        self,
        datasets: Iterable[Dataset],
    ):
        super().__init__(datasets=datasets)

    @property
    def df(self):
        "The concatenation of the two underlying metadata DataFrames."
        return super().df

    def subset(
        self, particpants_sessions: DataFrameType | Iterable[tuple[str, str]]
    ) -> Self:
        sub_datasets = []
        not_empty = False
        for dataset in self.datasets:
            try:
                sub_datasets.append(dataset.subset(particpants_sessions))
            except RuntimeError:  # empty dataset
                continue
            else:
                not_empty = True

        if not not_empty:
            raise RuntimeError(
                "No (participant, session) pairs are in the dataset. This would lead to an empty dataset!"
            )

        return type(self)(
            sub_datasets,
        )

    def get_sample_info(self, idx: int, column: str) -> Any:
        dataset_idx, idx_in_dataset = self._get_dataset_and_rank(idx)
        try:
            return self.datasets[dataset_idx].get_sample_info(idx_in_dataset, column)
        except KeyError as e:
            raise KeyError(
                f"No column named '{column}' in the metadata DataFrame of the dataset from which the sample is taken."
            ) from e

    def __len__(self) -> int:
        return int(np.sum([len(dataset) for dataset in self.datasets]))

    def __getitem__(self, idx: int) -> Sample:
        dataset_idx, idx_in_dataset = self._get_dataset_and_rank(idx)

        return self.datasets[dataset_idx][idx_in_dataset]

    def _get_dataset_and_rank(self, idx: int) -> tuple[int, int]:
        """
        Gets the dataset from which is the sample.
        """
        self._check_idx(idx)
        cum_len = np.cumsum([len(dataset) for dataset in self.datasets])
        dataset_idx = bisect_right(cum_len, idx)
        if dataset_idx > 0:
            idx_in_dataset = int(idx - cum_len[dataset_idx - 1])
        else:
            idx_in_dataset = idx

        return dataset_idx, idx_in_dataset

    @staticmethod
    def _merge_dfs(datasets: Sequence[Dataset]) -> pd.DataFrame:
        df: pd.DataFrame = pd.concat(
            [dataset.df for dataset in datasets],
            keys=range(len(datasets)),
            names=[DATASET_ID],
        )

        return df.reset_index(
            drop=False,
            level=DATASET_ID,
        ).reset_index(drop=True)
