from __future__ import annotations

from pathlib import Path

import pandas as pd

from clinicadl.data.datasets import CapsDataset
from clinicadl.dictionary.suffixes import TSV
from clinicadl.dictionary.words import DATA, SPLIT
from clinicadl.utils.typing import PathType

from ...base import Directory


class DataSplitDir(Directory):
    def __init__(self, parents_path: PathType, split: int):
        super().__init__(path=Path(parents_path) / f"{SPLIT}-{split}")

        self.df = None

    def _create(self, dataset: CapsDataset):
        super()._create()
        self.df = dataset.df
        self.df.to_csv(self.data_tsv, sep="\t", index=False)

    def load(self):
        super().load()
        try:
            self.df = pd.read_csv(self.data_tsv, sep="\t")
        except pd.errors.EmptyDataError:
            self.df = pd.DataFrame()

        # TODO : Add check for column and index ?

    @property
    def data_tsv(self) -> Path:
        return self.path / (DATA + TSV)
