# coding: utf8
# TODO: create a folder for generate/ prepare_data/ data to deal with capsDataset objects ?
import abc
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel
from torch.utils.data import Dataset

from clinicadl.dataset.caps_dataset import CapsDataset
from clinicadl.dataset.config.extraction import (
    ExtractionConfig,
    ExtractionImageConfig,
    ExtractionPatchConfig,
    ExtractionROIConfig,
    ExtractionSliceConfig,
)
from clinicadl.dataset.config.preprocessing import PreprocessingConfig
from clinicadl.dataset.config.utils import (
    get_preprocessing_and_mode_from_json,
)
from clinicadl.transforms.config import TransformsConfig
from clinicadl.utils.enum import (
    ExtractionMethod,
    Pattern,
    Preprocessing,
    SliceDirection,
    SliceMode,
    Template,
)
from clinicadl.utils.exceptions import (
    ClinicaDLCAPSError,
    ClinicaDLConcatError,
    ClinicaDLTSVError,
)
from clinicadl.utils.iotools.clinica_utils import check_caps_folder
from clinicadl.utils.iotools.utils import path_decoder, read_json

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

        self.caps_dict = self.compute_caps_dict()
        self.check_configs()

        self.eval_mode = False

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        for start, stop, dataset_index in self._indexes:
            if start <= index < stop:
                dataset = self._datasets[dataset_index]
                return dataset[index - start]

    def __len__(self) -> int:
        return self._len

    def check_configs(self):
        extraction = self._datasets[len(self._datasets) - 1].extraction
        preprocessing = self._datasets[len(self._datasets) - 1].preprocessing
        transforms = self._datasets[len(self._datasets) - 1].transforms
        size = self._datasets[len(self._datasets) - 1].size
        elem_per_image = self._datasets[len(self._datasets) - 1].elem_per_image

        for idx in range(len(self._datasets) - 1):
            if self._datasets[idx].extraction != extraction:
                raise ClinicaDLConcatError(
                    f"Different extraction modes found in datasets. "
                    f"Dataset {idx+1}: {self._datasets[idx].extraction}, "
                    f"Dataset {len(self._datasets)}: {extraction}"
                )

            if self._datasets[idx].preprocessing != preprocessing:
                raise ClinicaDLConcatError(
                    f"Different preprocessing modes found in datasets. "
                    f"Dataset {idx+1}: {self._datasets[idx].preprocessing}, "
                    f"Dataset {len(self._datasets)}: {preprocessing}"
                )

            if self._datasets[idx].transforms != transforms:
                raise ClinicaDLConcatError(
                    f"Different transforms modes found in datasets. "
                    f"Dataset {idx+1}: {self._datasets[idx].transforms}, "
                    f"Dataset {len(self._datasets)}: {transforms}"
                )
            if self._datasets[idx].size != size:
                raise ClinicaDLConcatError(
                    f"Different size modes found in datasets. "
                    f"Dataset {idx+1}: {self._datasets[idx].size}, "
                    f"Dataset {len(self._datasets)}: {size}"
                )
            if self._datasets[idx].elem_per_image != elem_per_image:
                raise ClinicaDLConcatError(
                    f"Different elem_per_image modes found in datasets. "
                    f"Dataset {idx+1}: {self._datasets[idx].elem_per_image}, "
                    f"Dataset {len(self._datasets)}: {elem_per_image}"
                )

        self.extraction = extraction
        self.preprocessing = preprocessing
        self.transforms = transforms
        self.size = size
        self.elem_per_image = elem_per_image

    def compute_caps_dict(self) -> Dict[str, Path]:
        caps_dict = dict()
        for idx in range(len(self._datasets)):
            cohort = idx
            caps_path = self._datasets[idx].caps_dict["caps_directory"]
            check_caps_folder(caps_path)
            caps_dict[cohort] = caps_path

        return caps_dict
