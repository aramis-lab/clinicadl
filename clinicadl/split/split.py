from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pydantic import Field, NonNegativeInt, field_validator
from typing_extensions import Self

from clinicadl.data.dataloader import DataLoader, DataLoaderConfig
from clinicadl.data.datasets import Dataset
from clinicadl.data.datasets.factory import get_dataset_from_dict
from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.objects import HasConfig

if TYPE_CHECKING:
    from clinicadl.data.dataloader import CollateFn


def _read_dataloader(
    serialized_loader: Optional[dict[str, Any]],
) -> Optional[DataLoaderConfig]:
    """
    To read the serialized dataloader, even if it is ``None``.
    """
    if serialized_loader:
        return DataLoaderConfig.from_dict(serialized_loader)
    return serialized_loader


class SplitConfig(ObjectConfig["Split"]):
    """Config class for ``Split``."""

    index: NonNegativeInt
    split_dir: Optional[Path]
    train_dataset: Dataset = Field(reader=get_dataset_from_dict)
    val_dataset: Dataset = Field(reader=get_dataset_from_dict)
    train_loader_config: Optional[DataLoaderConfig] = Field(
        default=None, reader=_read_dataloader
    )
    val_loader_config: Optional[DataLoaderConfig] = Field(
        default=None, reader=_read_dataloader
    )

    @field_validator("split_dir", mode="after")
    @classmethod
    def _check_split_dir(cls, v: Optional[Path]) -> Path:
        """
        Checks that the split dir exists.
        """
        if v:
            assert v.exists(), f"'split_dir' ({str(v)}) doesn't exist"

        return v

    @classmethod
    def _get_class(cls) -> type[Split]:
        return Split


class Split(HasConfig[SplitConfig]):
    """
    An object containing the relevant information on a split.

    More precisely, the dataclass contain the training and validation datasets, as well as
    the split index and the split directory used to split the dataset.

    Then, when :py:meth:`~Split.build_train_loader` and :py:meth:`build_val_loader` will be called,
    the training and validation :py:class:`~torch.utils.data.DataLoader` can be accessed.

    Finally, to instantiate Data Parallelism, that will distribute the training and validation sets
    across devices, the user can use :py:meth:`parallelism`.
    """

    _config_type = SplitConfig

    def __init__(
        self,
        index: int,
        train_dataset: Dataset,
        val_dataset: Dataset,
        split_dir: Optional[Path] = None,
    ):
        self.config = self._config_type(
            index=index,
            split_dir=split_dir,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
        self._dp_degree: Optional[int] = None
        self._rank: Optional[int] = None

    @property
    def index(self) -> int:
        """The index of the split."""
        return self.config.index

    @property
    def split_dir(self) -> Optional[Path]:
        """A potential split directory associated to this split."""
        return self.config.split_dir

    @property
    def train_dataset(self) -> Dataset:
        """The training set."""
        return self.config.train_dataset

    @property
    def val_dataset(self) -> Dataset:
        """The validation set."""
        return self.config.val_dataset

    @property
    def train_loader(self) -> DataLoader:
        """To access the training :py:class:`torch.utils.data.DataLoader`."""
        if not self.config.train_loader_config:
            raise RuntimeError(
                "The split has no training dataloader defined. Please run 'build_train_loader'"
            )
        return self.config.train_loader_config.get_object(
            dataset=self.train_dataset,
            dp_degree=self._dp_degree,
            rank=self._rank,
        )

    @property
    def val_loader(self) -> DataLoader:
        """To access the validation :py:class:`torch.utils.data.DataLoader`."""
        if not self.config.val_loader_config:
            raise RuntimeError(
                "The split has no validation dataloader defined. Please run 'build_val_loader'"
            )
        return self.config.val_loader_config.get_object(
            dataset=self.val_dataset,
            dp_degree=self._dp_degree,
            rank=self._rank,
        )

    def parallelism(self, dp_degree: int, rank: int) -> None:
        """
        Instantiates data parallelism. Training and validation sets will then be distributed
        across devices.

        Parameters
        ----------
        dp_degree : int
           The degree of data parallelism.
        rank : int
            Process id within the data parallelism communicator.

        Raises
        ------
        ValueError
            If ``rank`` is greater than ``dp_degree``.
        """
        if rank >= dp_degree:
            raise ValueError(
                "'rank' must be strictly smaller than 'dp_degree'. Got "
                f"dp_degree={dp_degree} and rank={rank}"
            )

        self._dp_degree = dp_degree
        self._rank = rank

    def build_train_loader(
        self,
        dataloader_config: Optional[DataLoaderConfig] = None,
        *,
        batch_size: int = 1,
        sampling_weights: Optional[str] = None,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        collate_fn: Optional[CollateFn] = None,
    ) -> None:
        """
        Builds a :py:class:`~torch.utils.data.DataLoader` for the training set of the split.

        Parameters
        ----------
        dataloader_config : Optional[DataLoaderConfig] (optional, default=None)
            A pre-configured :py:class:`~clinicadl.data.dataloader.DataLoaderConfig`.
            If passed, the arguments in this configuration object will prevail, otherwise
            the following arguments will be used.
        batch_size : int (optional, default=1)
            Batch size for the DataLoader. Used if ``dataloader_config`` is not provided.
        sampling_weights : Optional[str] (optional, default=None)
            Name of the column in the dataframe of the dataset where to find the sampling
            weights. The column must contain ``float`` values.

            The probability of sampling a certain sample is proportional to the associated value
            in this column of the dataframe.

            Used if ``dataloader_config`` is not provided.
        shuffle : bool (optional, default=True)
            Whether to shuffle the data.

            .. note::

                If ``sampling_weights`` is passed, the data will be fetched randomly with
                replacement, no matter the value of ``shuffle``.

            Used if ``dataloader_config`` is not provided.
        num_workers : int (optional, default=0)
            Number of workers for data loading. Used if ``dataloader_config`` is not provided.
        pin_memory : bool (optional, default=True)
            Whether to copy Tensors into device/CUDA pinned memory before returning them.
            Used if ``dataloader_config`` is not provided.
        drop_last : bool (optional, default=False)
            Whether to drop the last incomplete batch. Used if ``dataloader_config`` is not provided.
        prefetch_factor : Optional[int] (optional, default=None)
            Number of batches loaded in advance by each worker. Can't be passed if ``num_workers=0``.
            Used if ``dataloader_config`` is not provided.
        persistent_workers : bool (optional, default=False)
            Whether to maintain the worker processes alive at the end of an epoch.
            Can't be passed if ``num_workers=0``. Used if ``dataloader_config`` is not provided.
        collate_fn : Optional[CollateFn], default=None
            To customize the way samples are collated into batches. See :py:mod:`clinicadl.data.dataloader.collate`.

        Raises
        ------
        ValueError
            If ``prefetch_factor`` or ``persistent_workers`` is passed, but ``num_workers=0``.
        ValueError
            If the dataset is an :py:class:`~clinicadl.data.datasets.UnpairedDataset`,
            and ``sampling_weights`` is not ``None``.
        KeyError
            If ``sampling_weights`` is not ``None``, but there is no column named like
            ``sampling_weights`` in the dataframe of the dataset.
        ValueError
            If ``sampling_weights`` is not ``None`` and the associated column cannot
            be converted to float values.
        """
        if dataloader_config:
            self.config.train_loader_config = dataloader_config
        else:
            self.config.train_loader_config = DataLoaderConfig(
                batch_size=batch_size,
                sampling_weights=sampling_weights,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=collate_fn,
            )

    def build_val_loader(
        self,
        dataloader_config: Optional[DataLoaderConfig] = None,
        *,
        batch_size: int = 1,
        sampling_weights: Optional[str] = None,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        collate_fn: Optional[CollateFn] = None,
    ) -> None:
        """
        Builds a :py:class:`~torch.utils.data.DataLoader` for the validation set of the split.

        Parameters
        ----------
        dataloader_config : Optional[DataLoaderConfig] (optional, default=None)
            A pre-configured :py:class:`~clinicadl.data.dataloader.DataLoaderConfig`.
            If passed, the arguments in this configuration object will prevail, otherwise
            the following arguments will be used.
        batch_size : int (optional, default=1)
            Batch size for the DataLoader. Used if ``dataloader_config`` is not provided.
        sampling_weights : Optional[str] (optional, default=None)
            Name of the column in the dataframe of the dataset where to find the sampling
            weights. The column must contain ``float`` values.

            The probability of sampling a certain sample is proportional to the associated value
            in this column of the dataframe.

            Used if ``dataloader_config`` is not provided.
        shuffle : bool (optional, default=False)
            Whether to shuffle the data.

            .. note::

                If ``sampling_weights`` is passed, the data will be fetched randomly with
                replacement, no matter the value of ``shuffle``.

            Used if ``dataloader_config`` is not provided.
        num_workers : int (optional, default=0)
            Number of workers for data loading. Used if ``dataloader_config`` is not provided.
        pin_memory : bool (optional, default=True)
            Whether to copy Tensors into device/CUDA pinned memory before returning them.
            Used if ``dataloader_config`` is not provided.
        drop_last : bool (optional, default=False)
            Whether to drop the last incomplete batch. Used if ``dataloader_config`` is not provided.
        prefetch_factor : Optional[int] (optional, default=None)
            Number of batches loaded in advance by each worker. Can't be passed if ``num_workers=0``.
            Used if ``dataloader_config`` is not provided.
        persistent_workers : bool (optional, default=False)
            Whether to maintain the worker processes alive at the end of an epoch.
            Can't be passed if ``num_workers=0``. Used if ``dataloader_config`` is not provided.
        collate_fn : Optional[CollateFn], default=None
            To customize the way samples are collated into batches. See :py:mod:`clinicadl.data.dataloader.collate`.

        Raises
        ------
        ValueError
            If ``prefetch_factor`` or ``persistent_workers`` is passed, but ``num_workers=0``.
        ValueError
            If the dataset is an :py:class:`~clinicadl.data.datasets.UnpairedDataset`,
            and ``sampling_weights`` is not ``None``.
        KeyError
            If ``sampling_weights`` is not ``None``, but there is no column named like
            ``sampling_weights`` in the dataframe of the dataset.
        ValueError
            If ``sampling_weights`` is not ``None`` and the associated column cannot
            be converted to float values.
        """
        if dataloader_config:
            self.config.val_loader_config = dataloader_config
        else:
            self.config.val_loader_config = DataLoaderConfig(
                batch_size=batch_size,
                sampling_weights=sampling_weights,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=collate_fn,
            )

    @classmethod
    def _from_config(cls, config: SplitConfig) -> Self:
        split = cls(
            **config.to_raw_dict(exclude=["train_loader_config", "val_loader_config"])
        )
        if config.train_loader_config:
            split.build_train_loader(config.train_loader_config)
        if config.val_loader_config:
            split.build_val_loader(config.val_loader_config)

        return split
