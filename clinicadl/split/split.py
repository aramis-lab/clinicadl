from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pydantic import Field, NonNegativeInt, field_validator
from typing_extensions import Self

from clinicadl.data.dataloader.loader import DataLoader, DataLoaderConfig
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
    An object containing the data associated to a split.

    More precisely, the dataclass contains the training and validation :py:class:`~clinicadl.data.datasets.Dataset`, as well as
    the split index and the split directory used to split the dataset.

    Then, when :py:meth:`~Split.build_train_loader` and :py:meth:`build_val_loader` has been called,
    the training and validation :py:class:`~clinicadl.data.dataloader.DataLoader` can be accessed.
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
        """The training dataset."""
        return self.config.train_dataset

    @property
    def val_dataset(self) -> Dataset:
        """The validation dataset."""
        return self.config.val_dataset

    @property
    def train_loader(self) -> DataLoader:
        """The training dataloader."""
        if not self.config.train_loader_config:
            raise RuntimeError(
                "The split has no training dataloader defined. Please run 'build_train_loader'"
            )
        return self.config.train_loader_config.get_object(dataset=self.train_dataset)

    @property
    def val_loader(self) -> DataLoader:
        """The validation dataloader."""
        if not self.config.val_loader_config:
            raise RuntimeError(
                "The split has no validation dataloader defined. Please run 'build_val_loader'"
            )
        return self.config.val_loader_config.get_object(dataset=self.val_dataset)

    def build_train_loader(
        self,
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
        Builds a :py:class:`~clinicadl.data.dataloader.DataLoader` for the training set of the split.

        See :py:class:`~clinicadl.data.dataloader.DataLoader` for a description of the parameters.
        """
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
        Builds a :py:class:`~clinicadl.data.dataloader.DataLoader` for the validation set of the split.

        See :py:class:`~clinicadl.data.dataloader.DataLoader` for a description of the parameters.
        """
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
            split.config.train_loader_config = config.train_loader_config
        if config.val_loader_config:
            split.config.val_loader_config = config.val_loader_config

        return split
