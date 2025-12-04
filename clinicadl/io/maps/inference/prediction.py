from __future__ import annotations

from pathlib import Path

from clinicadl.dictionary.suffixes import TSV
from clinicadl.dictionary.words import CAPS, OUTPUT

from ...base import Directory
from .base import InferenceDir, InferenceGroupDir, InferenceSplitDir


class ModelDir(Directory):
    @property
    def caps_output(self) -> Path:
        return self.path / f"{CAPS}_{OUTPUT}"

    @property
    def output_tsv(self) -> Path:
        return (self.path / OUTPUT).with_suffix(TSV)

    def read(self) -> None:
        """
        Checks and reads the directory to find the files inside.

        Raises
        ------
        FileNotFoundError
            If an expected directory or file is missing.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Directory {str(self.path)} does not exist.")


class PredictionSplitDir(InferenceSplitDir[ModelDir]):
    _dir_type = ModelDir


class PredictionGroupDir(InferenceGroupDir[PredictionSplitDir]):
    _dir_type = PredictionSplitDir


class PredictionDir(InferenceDir[PredictionGroupDir]):
    _dir_type = PredictionGroupDir
