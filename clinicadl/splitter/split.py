from pathlib import Path
from typing import Optional

from pydantic import NonNegativeInt
from torch.utils.data import DataLoader

from clinicadl.dataset.datasets.caps_dataset import CapsDataset
from clinicadl.splitter.dataloader import DataLoaderConfig
from clinicadl.utils.config import ClinicaDLConfig

from .defaults import (
    BATCH_SIZE,
    DP_DEGREE,
    DROP_LAST,
    NUM_WORKERS,
    PIN_MEMORY,
    PREFETCH_FACTOR,
    RANK,
    SAMPLING_WEIGHTS,
    SHUFFLE,
)


class Split(ClinicaDLConfig):
    """Dataclass that contains all the useful info on the split."""

    index: NonNegativeInt
    split_dir: Path
    train_dataset: CapsDataset
    val_dataset: CapsDataset
    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None
    train_loader_config: Optional[DataLoaderConfig] = None
    val_loader_config: Optional[DataLoaderConfig] = None

    def build_train_loader(
        self,
        dataloader_config: Optional[DataLoaderConfig] = None,
        *,
        batch_size: int = BATCH_SIZE,
        sampling_weights: Optional[str] = SAMPLING_WEIGHTS,
        shuffle: bool = SHUFFLE,
        drop_last: bool = DROP_LAST,
        num_workers: int = NUM_WORKERS,
        prefetch_factor: Optional[int] = PREFETCH_FACTOR,
        pin_memory: bool = PIN_MEMORY,
        dp_degree: Optional[int] = DP_DEGREE,
        rank: Optional[int] = RANK,
    ) -> None:
        """
        Build a train loader for the split.

        Parameters
        ----------
        dataloader_config : Optional[DataLoaderConfig] (optional, default=None)
            Pre-configured DataLoader configuration.
        batch_size : int (optional, default=1)
            Batch size for the DataLoader (used if `dataloader_config` is not provided).
        sampling_weights : Optional[str] (optional, default=None)
            Name of the column in the dataframe of the CapsDatasets where to find the sampling
            weights (used if `dataloader_config` is not provided).
        shuffle : bool (optional, default=False)
            Whether to shuffle the data (used if `dataloader_config` is not provided).
        drop_last : bool (optional, default=False)
            Whether to drop the last incomplete batch (used if `dataloader_config` is not provided).
        num_workers : int (optional, default=0)
            Number of workers for data loading (used if `dataloader_config` is not provided).
        prefetch_factor : Optional[int] (optional, default=None)
            Prefetch factor if num_workers is not 0 (used if `dataloader_config` is not provided).
        dp_degree : Optional[int] (optional, default=None)
           The degree of data parallelism. None if no data parallelism.
        rank : Optional[int] (optional, default=None)
            Process id within the data parallelism communicator.
            None if no data parallelism.

        Raises
        ------
        ValueError
            If one of 'dp_degree' and 'rank' is None but the other is not None.
        KeyError
            If 'sampling_weights' is passed but there is no such column in the dataframe
            of the train dataset.
        """
        if dataloader_config:
            self.train_loader_config = dataloader_config
        else:
            self.train_loader_config = DataLoaderConfig(
                batch_size=batch_size,
                sampling_weights=sampling_weights,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
            )
        self.train_loader = self.train_loader_config.get_dataloader(
            dataset=self.train_dataset,
            dp_degree=dp_degree,
            rank=rank,
        )

    def build_val_loader(
        self,
        dataloader_config: Optional[DataLoaderConfig] = None,
        *,
        batch_size: int = BATCH_SIZE,
        sampling_weights: Optional[str] = SAMPLING_WEIGHTS,
        shuffle: bool = SHUFFLE,
        drop_last: bool = DROP_LAST,
        num_workers: int = NUM_WORKERS,
        prefetch_factor: Optional[int] = PREFETCH_FACTOR,
        pin_memory: bool = PIN_MEMORY,
        dp_degree: Optional[int] = DP_DEGREE,
        rank: Optional[int] = RANK,
    ) -> None:
        """
        Build a validation loader for the split.

        Parameters
        ----------
        dataloader_config : Optional[DataLoaderConfig] (optional, default=None)
            Pre-configured DataLoader configuration.
        batch_size : int (optional, default=1)
            Batch size for the DataLoader (used if `dataloader_config` is not provided).
        sampling_weights : Optional[str] (optional, default=None)
            Name of the column in the dataframe of the CapsDatasets where to find the sampling
            weights (used if `dataloader_config` is not provided).
        shuffle : bool (optional, default=False)
            Whether to shuffle the data (used if `dataloader_config` is not provided).
        drop_last : bool (optional, default=False)
            Whether to drop the last incomplete batch (used if `dataloader_config` is not provided).
        num_workers : int (optional, default=0)
            Number of workers for data loading (used if `dataloader_config` is not provided).
        prefetch_factor : Optional[int] (optional, default=None)
            Prefetch factor if num_workers is not 0 (used if `dataloader_config` is not provided).
        dp_degree : Optional[int] (optional, default=None)
           The degree of data parallelism. None if no data parallelism.
        rank : Optional[int] (optional, default=None)
            Process id within the data parallelism communicator.
            None if no data parallelism.

        Raises
        ------
        ValueError
            If one of 'dp_degree' and 'rank' is None but the other is not None.
        KeyError
            If 'sampling_weights' is passed but there is no such column in the dataframe
            of the validation dataset.
        """
        if dataloader_config:
            self.val_loader_config = dataloader_config
        else:
            self.val_loader_config = DataLoaderConfig(
                batch_size=batch_size,
                sampling_weights=sampling_weights,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
            )
        self.val_loader = self.val_loader_config.get_dataloader(
            dataset=self.val_dataset,
            dp_degree=dp_degree,
            rank=rank,
        )
