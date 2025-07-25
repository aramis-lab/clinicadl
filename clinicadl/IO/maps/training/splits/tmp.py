from __future__ import annotations

from pathlib import Path

from clinicadl.dictionary.suffixes import PTH, TAR
from clinicadl.dictionary.words import (
    MODEL,
    OPTIMIZER,
    TMP,
)
from clinicadl.utils.typing import PathType

from ...base import Directory


class TmpDir(Directory):
    def __init__(self, parent_dir: PathType):
        super().__init__(path=Path(parent_dir) / TMP)
        pass

    @property
    def model(self) -> Path:
        return (self.path / MODEL).with_suffix(PTH + TAR)

    @property
    def optimizer(self) -> Path:
        return (self.path / OPTIMIZER).with_suffix(PTH + TAR)

    def remove(self) -> None:
        """Removes the temporary files."""
        if self.model.is_file():
            self.model.unlink()
        if self.optimizer.is_file():
            self.optimizer.unlink()
        if self.path.is_dir():
            self.path.rmdir()
