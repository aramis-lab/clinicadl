from pathlib import Path
from typing import Optional

from pydantic import NonNegativeInt, PositiveInt
from torch.utils.data import DataLoader

from clinicadl.data.dataloader.config import DataLoaderConfig
from clinicadl.data.datasets.caps_dataset import CapsDataset
from clinicadl.utils.config import ClinicaDLConfig


class Split(ClinicaDLConfig):
    """
    Dataclass that contains all the useful information on a split.

    Parameters
    ----------
    index : NonNegativeInt
        The index of the split.
    split_dir : Path
        Directory from which the split was built.
    train_dataset : CapsDataset
        The training CapsDataset.
    val_dataset : CapsDataset
        The validation CapsDataset.
    train_loader : Optional[DataLoader]
        The training PyTorch DataLoader. Will be None until `build_train_loader`
        is called.
    val_loader : Optional[DataLoader]
        The validation PyTorch DataLoader. Will be None until `build_val_loader`
        is called.
    train_loader_config : Optional[DataLoaderConfig]
        A dataclass saving the parameters used when calling `build_train_loader`.
        For reproducibility.
    val_loader_config : Optional[DataLoaderConfig]
        A dataclass saving the parameters used when calling `build_val_loader`.
        For reproducibility.
    """

    index: NonNegativeInt
    split_dir: Path
    train_dataset: CapsDataset
    val_dataset: CapsDataset
    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None
    train_loader_config: Optional[DataLoaderConfig] = None
    val_loader_config: Optional[DataLoaderConfig] = None
    _dp_degree: Optional[PositiveInt] = None
    _rank: Optional[NonNegativeInt] = None

    def reset(self) -> None:
        """
        Resets the computed fields of the Split object
        ('train_loader', 'val_loader', etc.).
        """
        self.train_loader = None
        self.val_loader = None
        self.train_loader_config = None
        self.val_loader_config = None
        self._dp_degree = None
        self._rank = None

    def parallelism(self, dp_degree: int, rank: int) -> None:
        """
        Instantiates data parallelism. The data will then be split
        across devices.

        Parameters
        ----------
        dp_degree : Optional[int] (optional, default=None)
           The degree of data parallelism.
        rank : Optional[int] (optional, default=None)
            Process id within the data parallelism communicator.
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
    ) -> None:
        """
        Builds a train loader for the split.

        Parameters
        ----------
        dataloader_config : Optional[DataLoaderConfig] (optional, default=None)
            Pre-configured DataLoader configuration.
        batch_size : int (optional, default=1)
            Batch size for the DataLoader (used if `dataloader_config` is not provided).
        sampling_weights : Optional[str] (optional, default=None)
            Name of the column in the dataframe of the CapsDataset where to find the sampling
            weights (used if `dataloader_config` is not provided). The column must contain
            float values.
        shuffle : bool (optional, default=True)
            Whether to shuffle the data (used if `dataloader_config` is not provided).
            .. note:: If `sampling_weights` is passed, the data will be fetched randomly with
            replacement. So, data are shuffled, no matter the argument `shuffle`.
        num_workers : int (optional, default=0)
            Number of workers for data loading (used if `dataloader_config` is not provided).
        pin_memory : bool (optional, default=True)
            whether to copy Tensors into device/CUDA pinned memory before returning them
            (used if `dataloader_config` is not provided).
        drop_last : bool (optional, default=False)
            Whether to drop the last incomplete batch (used if `dataloader_config` is not provided).
        prefetch_factor : Optional[int] (optional, default=None)
            Number of batches loaded in advance by each worker (used if `dataloader_config` is not provided).
            Can't be passed if `num_workers` is 0.
        persistent_workers : bool (optional, default=False)
            Whether to maintain the worker processes alive at the end of an epoch (used if `dataloader_config` is not provided).
            Can't be passed if `num_workers` is 0.

        Raises
        ------
        ValueError
            If `prefetch_factor` or `persistent_workers` is passed, but `num_workers` is 0.
        KeyError
            If `sampling_weights` is passed but there is no such column in the dataframe
            of the train dataset.
        KeyError
            If the column passed in `sampling_weights` cannot be converted to floats.
        """

        self.train_dataset._count_samples()
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
                persistent_workers=persistent_workers,
            )
        self.train_loader = self.train_loader_config.get_dataloader(
            dataset=self.train_dataset,
            dp_degree=self._dp_degree,
            rank=self._rank,
        )
        self.train_loader.type = "train"

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
    ) -> None:
        """
        Builds a validation loader for the split.

        Parameters
        ----------
        dataloader_config : Optional[DataLoaderConfig] (optional, default=None)
            Pre-configured DataLoader configuration.
        batch_size : int (optional, default=1)
            Batch size for the DataLoader (used if `dataloader_config` is not provided).
        sampling_weights : Optional[str] (optional, default=None)
            Name of the column in the dataframe of the CapsDataset where to find the sampling
            weights (used if `dataloader_config` is not provided). The column must contain
            float values.
        shuffle : bool (optional, default=False)
            Whether to shuffle the data (used if `dataloader_config` is not provided).
            .. note:: If `sampling_weights` is passed, the data will be fetched randomly with
            replacement. So, data are shuffled, no matter the argument `shuffle`.
        num_workers : int (optional, default=0)
            Number of workers for data loading (used if `dataloader_config` is not provided).
        pin_memory : bool (optional, default=True)
            whether to copy Tensors into device/CUDA pinned memory before returning them
            (used if `dataloader_config` is not provided).
        drop_last : bool (optional, default=False)
            Whether to drop the last incomplete batch (used if `dataloader_config` is not provided).
        prefetch_factor : Optional[int] (optional, default=None)
            Number of batches loaded in advance by each worker (used if `dataloader_config` is not provided).
            Can't be passed if `num_workers` is 0.
        persistent_workers : bool (optional, default=False)
            Whether to maintain the worker processes alive at the end of an epoch (used if `dataloader_config` is not provided).
            Can't be passed if `num_workers` is 0.

        Raises
        ------
        ValueError
            If `prefetch_factor` or `persistent_workers` is passed, but `num_workers` is 0.
        KeyError
            If `sampling_weights` is passed but there is no such column in the dataframe
            of the validation dataset.
        KeyError
            If the column passed in `sampling_weights` cannot be converted to floats.
        """
        self.val_dataset._count_samples()
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
                persistent_workers=persistent_workers,
            )
        self.val_loader = self.val_loader_config.get_dataloader(
            dataset=self.val_dataset,
            dp_degree=self._dp_degree,
            rank=self._rank,
        )
        self.val_loader.type = "val"
