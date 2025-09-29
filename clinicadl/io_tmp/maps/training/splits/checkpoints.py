from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from clinicadl.dictionary.suffixes import PTH, TAR
from clinicadl.dictionary.words import (
    CHECKPOINTS,
    EPOCH,
    MODEL,
)
from clinicadl.utils.typing import PathType

from ...base import Directory


class EpochDir(Directory):
    def __init__(self, parents_path: PathType, epoch: int):
        super().__init__(path=Path(parents_path) / f"{EPOCH}-{epoch}")
        self.epoch = epoch

    def load(self):
        super().load()
        # TODO : add check ?

    @property
    def model(self) -> Path:
        return self.path / (MODEL + PTH + TAR)


class CheckpointsDir(Directory):
    _epoch_dir_type = EpochDir

    def __init__(self, parents_path: PathType):
        super().__init__(path=Path(parents_path) / CHECKPOINTS)

        self.epochs: Dict[int, EpochDir] = {}

    def _create_epoch(self, epoch: int):
        epoch_dir = self._epoch_dir_type(parents_path=self.path, epoch=epoch)
        epoch_dir._create(_exists_ok=True)
        self.epochs[epoch] = epoch_dir

    def clear(self, except_epoch: Optional[int] = None) -> None:
        for epoch, dir_ in self.epochs.items():
            if epoch != except_epoch:
                dir_.remove()

    def load(self):
        super().load()

        for epoch in self.epoch_list:
            epoch_dir = self._epoch_dir_type(parents_path=self.path, epoch=epoch)
            epoch_dir.load()
            self.epochs[epoch] = epoch_dir

    @property
    def epoch_list(self):
        if self.is_empty():
            return []
        return [
            int(x.name.split("-")[1])
            for x in self.path.iterdir()
            if x.is_dir() and x.name.startswith(EPOCH)
        ]
