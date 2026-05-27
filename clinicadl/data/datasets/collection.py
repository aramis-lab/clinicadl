from abc import abstractmethod
from typing import Any, Generic, Iterable, Sequence, TypeVar

import pandas as pd
from pydantic import Field
from typing_extensions import Self

from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.dictionary.words import DATASET_ID
from clinicadl.utils.objects import HasConfig
from clinicadl.utils.typing import DataFrameType

from .base import Dataset

D = TypeVar("D", bound=Dataset)


def _get_dataset_from_dict(data: dict[str, Any]) -> Dataset:
    """
    To avoid circular imports.
    """
    from .factory import get_dataset_from_dict

    return get_dataset_from_dict(data)


class CollectionDatasetConfig(ObjectConfig["CollectionDataset"]):
    """
    Base config class for ``CollectionDatasets``.
    """

    datasets: tuple[Dataset, ...] = Field(
        reader=lambda x: tuple(map(_get_dataset_from_dict, x))
    )

    @classmethod
    def _check_dataset_id_column(
        cls, datasets: tuple[Dataset, ...]
    ) -> tuple[Dataset, ...]:
        for dataset in datasets:
            assert (
                DATASET_ID not in dataset.df.columns
            ), f"'{DATASET_ID}' is a protected name. It cannot be in the DataFrames of the underlying datasets."

        return datasets

    @classmethod
    def _check_at_least_two_datasets(
        cls, datasets: tuple[Dataset, ...]
    ) -> tuple[Dataset, ...]:
        assert (
            len(datasets) >= 2
        ), f"{cls._get_name()} requires at least 2 datasets to join!"

        return datasets


class CollectionDataset(HasConfig[CollectionDatasetConfig], Dataset, Generic[D]):
    """
    Abstract class defining some common logic for the datasets
    that are a collections of :py:class:`~clinicadl.data.datasets.Dataset`.
    """

    def __init__(self, datasets: Iterable[D], **kwargs):
        self.config = self._config_type(datasets=datasets, **kwargs)
        self._df = self._merge_dfs(self.config.datasets)
        self.datasets = self.config.datasets

    def eval(self) -> None:
        for dataset in self.datasets:
            dataset.eval()

    def train(self) -> None:
        for dataset in self.datasets:
            dataset.train()

    def subset(
        self, particpants_sessions: DataFrameType | Iterable[tuple[str, str]]
    ) -> Self:
        return type(self)(
            list(dataset.subset(particpants_sessions) for dataset in self.datasets),
            **self.config.to_dict(exclude=["datasets", "name_"]),
        )

    @staticmethod
    @abstractmethod
    def _merge_dfs(datasets: Sequence[D]) -> pd.DataFrame:
        """
        Merges the dataframes from all the datasets.
        """
