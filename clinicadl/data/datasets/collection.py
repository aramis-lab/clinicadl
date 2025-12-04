from abc import abstractmethod
from typing import Any, Iterable, Sequence, Union

import pandas as pd
from pydantic import Field, field_validator
from typing_extensions import Self

from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.exceptions import TensorConversionError
from clinicadl.utils.objects import HasConfig
from clinicadl.utils.typing import DataFrameType

from .abstract import ClinicaDLDataset
from .multi_samples import MultiSamplesDataset
from .tensor import TensorDataset


def _get_dataset_from_dict(data: dict[str, Any]) -> ClinicaDLDataset:
    """
    To avoid circular imports.
    """
    from .factory import get_dataset_from_dict

    return get_dataset_from_dict(data)


class CollectionDatasetConfig(ObjectConfig["CollectionDataset"]):
    """
    Base config class for ``CollectionDatasets``.
    """

    datasets: tuple[MultiSamplesDataset, ...] = Field(
        reader=lambda x: tuple(map(_get_dataset_from_dict, x))
    )

    @field_validator("datasets", mode="after")
    @classmethod
    def _check_n_samples(
        cls, datasets: Sequence[MultiSamplesDataset]
    ) -> Sequence[MultiSamplesDataset]:
        """
        Checks that tensor conversion has been performed before joining the dataset.
        """
        for dataset in datasets:
            if isinstance(dataset, TensorDataset):
                if not dataset.converted:
                    raise TensorConversionError(
                        "Tensor conversion must be performed BEFORE joining the datasets. Please call "
                        "'to_tensors' or 'read_tensor_conversion' for each dataset."
                    )
            else:
                if not dataset._has_len:
                    raise ValueError(
                        f"{cls._get_name()} needs the number of samples per image for each underlying dataset. "
                        "Please ensure that 'n_samples' is a column of the metadata DataFrame of the datasets."
                    )

        return datasets


class CollectionDataset(HasConfig[CollectionDatasetConfig], ClinicaDLDataset):
    """
    Abstract class defining some common logic for the :py:class:`~clinicadl.data.datasets.ClinicaDLDataset`
    that are a collections of :py:class:`~clinicadl.data.datasets.MultiSamplesDataset`.
    """

    datasets: tuple[MultiSamplesDataset, ...]
    _df: pd.DataFrame

    def __init__(self, datasets: Iterable[MultiSamplesDataset], **kwargs):
        self.config = self._config_type(datasets=datasets, **kwargs)
        self._df = self._merge_dfs(self.config.datasets)
        self.datasets = self.config.datasets

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def eval(self) -> None:
        for dataset in self.datasets:
            dataset.eval()

    def train(self) -> None:
        for dataset in self.datasets:
            dataset.train()

    def subset(
        self, particpants_sessions: Union[DataFrameType, Sequence[tuple[str, str]]]
    ) -> Self:
        return type(self)(
            list(dataset.subset(particpants_sessions) for dataset in self.datasets)
        )

    def describe(self) -> tuple[dict[str, Any], ...]:
        """
        Returns a description of the underlying datasets.

        Returns
        -------
        tuple[dict[str, Any], ...]
            The descriptions returned by :py:meth:`ClinicaDLDataset.describe
            <clinicadl.data.datasets.ClinicaDLDataset.describe>` for each
            sub-dataset forming the current dataset.
        """
        return tuple([dataset.describe() for dataset in self.datasets])

    @staticmethod
    @abstractmethod
    def _merge_dfs(datasets: Sequence[MultiSamplesDataset]) -> pd.DataFrame:
        """
        Merges the dataframes from all the datasets.
        """
