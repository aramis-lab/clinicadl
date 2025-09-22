from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from clinicadl.dictionary.suffixes import PTH, TAR
from clinicadl.dictionary.words import (
    CALLBACKS,
    MODEL,
    TMP,
)
from clinicadl.utils.typing import PathType

from ...base import Directory


class TmpDir(Directory):
    def __init__(self, parent_dir: PathType, epoch: Optional[int] = None):
        path = Path(parent_dir) / TMP
        if epoch is not None:
            path /= f"epoch-{epoch}"
        super().__init__(path=path)

    @property
    def model(self) -> Path:
        return (self.path / MODEL).with_suffix(PTH + TAR)

    @property
    def callbacks(self) -> Path:
        if not (self.path / CALLBACKS).is_dir():
            (self.path / CALLBACKS).mkdir(parents=True)
        return self.path / CALLBACKS

    def remove(self) -> None:
        """Removes the temporary files."""
        shutil.rmtree(self.path)

    def clear(self, except_epoch: Optional[int] = None) -> None:
        """Removes the temporary files."""
        list_dir = [
            d
            for d in self.path.iterdir()
            if d.is_dir() and d.name != f"epoch-{except_epoch}"
        ]
        for dir in list_dir:
            shutil.rmtree(dir)
