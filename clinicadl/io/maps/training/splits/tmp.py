from __future__ import annotations

from pathlib import Path
from typing import Optional

from clinicadl.dictionary.suffixes import JSON, PTH, TAR
from clinicadl.dictionary.words import (
    CALLBACKS,
    METRICS,
    MODEL,
    SCALER,
    STATE,
    VALIDATION,
)

from ....base import Directory
from ...metrics import MetricsDir
from ...utils import EpochsDir


class EpochTmpDir(Directory):
    def __init__(self, path: Path):
        super().__init__(path)
        self._validation_metrics = MetricsDir(
            path=self.path / f"{VALIDATION}_{METRICS}"
        )

    @property
    def callbacks(self) -> Path:
        return self.path / CALLBACKS

    @property
    def validation_metrics(self) -> MetricsDir:
        return self._validation_metrics

    @property
    def model(self) -> Path:
        return (self.path / MODEL).with_suffix(PTH + TAR)

    @property
    def scaler(self) -> Path:
        return (self.path / SCALER).with_suffix(JSON)

    @property
    def state(self) -> Path:
        return (self.path / STATE).with_suffix(JSON)


class TmpDir(EpochsDir[EpochTmpDir]):
    _dir_type = EpochTmpDir

    def clear(self, except_epoch: Optional[int] = None) -> None:
        """
        Clears the tmp directory by removing all epochs
        except ``except_epoch``.

        Parameters
        ----------
        except_epoch : Optional[int], default=None
            The checkpoint to keep. If ``None``, the whole directory will be cleared.
        """
        self._find_item_dirs()
        for epoch in self.epochs_list:
            if epoch != except_epoch:
                self.epochs[epoch].remove(non_empty_ok=True)
        self._find_item_dirs()
