from __future__ import annotations

from pathlib import Path
from typing import Dict

from clinicadl.dictionary.suffixes import PTH, TAR
from clinicadl.dictionary.words import (
    CHECKPOINTS,
    EPOCH,
    MODEL,
    OPTIMIZER,
)
from clinicadl.utils.typing import PathType

from ...base import Directory


class EpochDir(Directory):
    def __init__(self, parents_path: PathType, epoch: int):
        super().__init__(path=Path(parents_path) / f"{EPOCH}-{epoch}")

    def load(self):
        super().load()
        # TODO : add check ?

    @property
    def model(self) -> Path:
        return self.path / (MODEL + PTH + TAR)

    @property
    def optimizer(self) -> Path:
        return self.path / (OPTIMIZER + PTH + TAR)


class CheckpointsDir(Directory):
    def __init__(self, parents_path: PathType):
        super().__init__(path=Path(parents_path) / CHECKPOINTS)

        self.epochs: Dict[int, EpochDir] = {}

    def _create_epoch(self, epoch: int):
        epoch_dir = EpochDir(parents_path=self.path, epoch=epoch)
        epoch_dir._create()
        self.epochs[epoch] = epoch_dir

    def load(self):
        super().load()

        for epoch in self.epoch_list:
            epoch_dir = EpochDir(parents_path=self.path, epoch=epoch)
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
