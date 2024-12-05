from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple, Union

from pydantic import NonNegativeInt
from torch.utils.data import DataLoader

from clinicadl.dataset.datasets.caps_dataset import CapsDataset

from .new_splitter.dataloader import get_dataloader
from .new_splitter.subset import Subset


@dataclass
class Split:
    """Dataclass that contains all the useful info on the split."""

    index: int
    train_dataset: CapsDataset
    val_dataset: CapsDataset
    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None

    def build_train_loader(
        self,
        batch_size: int,
        sampling_weights: Optional[str] = None,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
        prefetch_factor: Optional[NonNegativeInt] = None,
        dp_degree: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        self.train_loader = get_dataloader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            sampling_weights=sampling_weights,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            dp_degree=dp_degree,
            rank=rank,
        )

    def build_val_loader(
        self,
        batch_size: int,
        sampling_weights: Optional[str] = None,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
        prefetch_factor: Optional[NonNegativeInt] = None,
        dp_degree: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        self.val_loader = get_dataloader(
            dataset=self.val_dataset,
            batch_size=batch_size,
            sampling_weights=sampling_weights,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            dp_degree=dp_degree,
            rank=rank,
        )
