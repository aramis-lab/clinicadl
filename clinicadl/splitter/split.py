from dataclasses import dataclass
from typing import Optional

from pydantic import ConfigDict, NonNegativeInt
from torch.utils.data import DataLoader

from clinicadl.dataset.datasets.caps_dataset import CapsDataset
from clinicadl.splitter.dataloader import DataLoaderConfig


@dataclass
class Split:
    """Dataclass that contains all the useful info on the split."""

    index: int
    train_dataset: CapsDataset
    val_dataset: CapsDataset
    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    def build_train_loader(
        self,
        dataloader_config: Optional[DataLoaderConfig] = None,
        *,
        batch_size: int = 1,
        sampling_weights: Optional[str] = None,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
        prefetch_factor: Optional[NonNegativeInt] = None,
        dp_degree: int = 1,
        rank: int = 1,
    ) -> None:
        """
        Build a train loader for the split.

        Parameters
        ----------
        dataloader_config : Optional[DataLoaderConfig], default=None
            Pre-configured DataLoader configuration.
        batch_size : Optional[int], default=None
            Batch size for the DataLoader (used if `dataloader_config` is not provided).
        sampling_weights : Optional[str], default=None
            Sampling weights for the DataLoader.
        shuffle : bool, default=False
            Whether to shuffle the data.
        num_workers : int, default=0
            Number of workers for data loading.
        drop_last : bool, default=False
            Whether to drop the last incomplete batch.
        prefetch_factor : Optional[NonNegativeInt], default=None
            Prefetch factor for the DataLoader.
        dp_degree : int, default=1
            Data parallelism degree.
        rank : int, default=1
            Rank for distributed data loading.

        Raises
        ------
        ValueError
            If neither a configuration object nor batch_size is provided.
        """
        if dataloader_config:
            self.train_loader = dataloader_config.get_dataloader(
                dataset=self.train_dataset
            )
        elif batch_size is not None:
            dataloader = DataLoaderConfig(
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
                prefetch_factor=prefetch_factor,
            )
            self.train_loader = dataloader.get_dataloader(
                dataset=self.train_dataset,
                sampling_weights=sampling_weights,
                dp_degree=dp_degree,
                rank=rank,
            )
        else:
            raise ValueError(
                "Either a DataLoaderConfig or batch_size must be provided."
            )

    def build_val_loader(
        self,
        dataloader_config: Optional[DataLoaderConfig] = None,
        *,
        batch_size: int = 1,
        sampling_weights: Optional[str] = None,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
        prefetch_factor: Optional[NonNegativeInt] = None,
        dp_degree: int = 1,
        rank: int = 1,
    ) -> None:
        """
        Build a train loader for the split.

        Parameters
        ----------
        dataloader_config : Optional[DataLoaderConfig], default=None
            Pre-configured DataLoader configuration.
        batch_size : Optional[int], default=None
            Batch size for the DataLoader (used if `dataloader_config` is not provided).
        sampling_weights : Optional[str], default=None
            Sampling weights for the DataLoader.
        shuffle : bool, default=False
            Whether to shuffle the data.
        num_workers : int, default=0
            Number of workers for data loading.
        drop_last : bool, default=False
            Whether to drop the last incomplete batch.
        prefetch_factor : Optional[NonNegativeInt], default=None
            Prefetch factor for the DataLoader.
        dp_degree : int, default=1
            Data parallelism degree.
        rank : int, default=1
            Rank for distributed data loading.

        Raises
        ------
        ValueError
            If neither a configuration object nor batch_size is provided.
        """
        if dataloader_config:
            self.val_loader = dataloader_config.get_dataloader(dataset=self.val_dataset)
        elif batch_size is not None:
            dataloader = DataLoaderConfig(
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
                prefetch_factor=prefetch_factor,
            )
            self.val_loader = dataloader.get_dataloader(
                dataset=self.val_dataset,
                sampling_weights=sampling_weights,
                dp_degree=dp_degree,
                rank=rank,
            )
        else:
            raise ValueError(
                "Either a DataLoaderConfig or batch_size must be provided."
            )
