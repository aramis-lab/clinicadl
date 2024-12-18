# coding: utf8
import abc
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from clinicadl.data.datasets import CapsDataset
from clinicadl.data.utils import CapsDatasetSample

logger = getLogger("clinicadl")


class ConcatDataset(CapsDataset):
    def __init__(self, datasets: List[CapsDataset]):
        self._datasets = datasets
        self._len = sum(len(dataset) for dataset in datasets)
        self._indexes = []

        # Calculate distribution of indexes in all datasets
        cumulative_index = 0
        for idx, dataset in enumerate(datasets):
            next_cumulative_index = cumulative_index + len(dataset)
            self._indexes.append((cumulative_index, next_cumulative_index, idx))
            cumulative_index = next_cumulative_index

        logger.debug(f"Datasets summary length: {self._len}")
        logger.debug(f"Datasets indexes: {self._indexes}")

        self.check_extraction()

        self.eval_mode = False

    def __getitem__(self, index: int) -> Optional[CapsDatasetSample]:
        for start, stop, dataset_index in self._indexes:
            if start <= index < stop:
                dataset = self._datasets[dataset_index]
                return dataset[index - start]

    def __len__(self) -> int:
        return self._len

    def check_extraction(self):
        extractions = [d.extraction for d in self._datasets]
        if all(
            i == extractions[0] for i in extractions
        ):  # check that all the CaspDataset have the same mode
            self.extraction = extractions[0]
        else:
            raise AttributeError(
                "All the CapsDataset must have the same extraction method: 'image','patch','roi','slice', etc."
            )
