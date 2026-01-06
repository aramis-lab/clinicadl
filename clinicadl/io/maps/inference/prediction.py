from __future__ import annotations

from pathlib import Path

from clinicadl.utils.dictionary.suffixes import TSV
from clinicadl.utils.dictionary.words import CAPS, OUTPUT

from ...base import Directory
from .base import (
    InferenceDir,
    InferenceGroupDir,
    InferenceResultsDir,
    InferenceSplitDir,
)


class PredictionModelDir(Directory):
    @property
    def caps_output(self) -> Path:
        return self.path / f"{CAPS}_{OUTPUT}"

    @property
    def output_tsv(self) -> Path:
        return (self.path / OUTPUT).with_suffix(TSV)


class PredictionSplitDir(InferenceSplitDir[PredictionModelDir]):
    _dir_type = PredictionModelDir


class PredictionResultsDir(InferenceResultsDir[PredictionSplitDir]):
    _dir_type = PredictionSplitDir


class PredictionGroupDir(InferenceGroupDir[PredictionResultsDir]):
    _results_dir_type = PredictionResultsDir


class PredictionDir(InferenceDir[PredictionGroupDir]):
    _dir_type = PredictionGroupDir
