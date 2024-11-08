import json
from logging import getLogger
from pathlib import Path
from typing import Optional, Tuple, Union

import nibabel as nib
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch import save as save_tensor

from clinicadl.dataset.config.extraction import (
    ALL_EXTRACTION_TYPES,
    ExtractionConfig,
    ExtractionImageConfig,
)
from clinicadl.dataset.config.preprocessing import (
    ALL_PREPROCESSING_TYPES,
    CustomPreprocessingConfig,
    DTIPreprocessingConfig,
    PETPreprocessingConfig,
    PreprocessingConfig,
)
from clinicadl.dataset.config.utils import (
    get_infos_from_json,
    get_preprocessing,
)
from clinicadl.dataset.datasets.caps_dataset import CapsDataset
from clinicadl.dataset.datasets.concat import ConcatDataset
from clinicadl.dataset.transforms.transforms import Transforms
from clinicadl.utils.enum import (
    DTIMeasure,
    DTISpace,
    Preprocessing,
    SUVRReferenceRegions,
    Tracer,
)
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
    ClinicaDLTSVError,
)
from clinicadl.utils.iotools.clinica_utils import (
    check_caps_folder,
    clinicadl_file_reader,
    container_from_filename,
    create_subs_sess_list,
    determine_caps_or_bids,
    get_subject_session_list,
)
from clinicadl.utils.iotools.utils import path_encoder

logger = getLogger("clinicadl.caps_reader")


class Reader:
    """Base reader class for BIDS and CAPS directories.

    Args:
        input_dir (Path): Path to the BIDS or CAPS directory.
        bids (bool): Flag indicating if the input is a BIDS directory.
    """

    def __init__(self, input_dir: Path, bids: bool) -> None:
        self.input_directory = input_dir
        self.bids = bids

    def preprocessing_folder(
        self, subject: str, session: str, preprocessing: Preprocessing
    ) -> Path:
        return (
            self.input_directory
            / "subjects"
            / subject
            / session
            / (preprocessing.value).replace("-", "_")
        )

    def get_preprocessing(
        self, preprocessing: Union[str, Preprocessing]
    ) -> PreprocessingConfig:
        """Get preprocessing configuration for the input directory.

        Args:
            preprocessing (Union[str, PreprocessingConfig]): Preprocessing type as a string or PreprocessingConfig.

        Returns:
            PreprocessingConfig: The configuration for preprocessing.
        """

        preprocessing_ = Preprocessing(preprocessing)
        subjects, sessions = get_subject_session_list(
            input_dir=self.input_directory, is_bids_dir=self.bids
        )
        if self.preprocessing_folder(
            subject=subjects[0], session=sessions[0], preprocessing=preprocessing_
        ).is_dir():
            preprocessing_config = get_preprocessing(preprocessing_)()
            preprocessing_config.from_bids = self.bids
            pattern = preprocessing_config.file_type.pattern

            def get_value(enum, pattern: str):
                for value in enum:
                    if value.value in pattern:
                        return value
                raise ValueError(
                    f"Could not match pattern '{pattern}' in {[e.value for e in enum]}"
                )

            if isinstance(preprocessing_config, PETPreprocessingConfig):
                preprocessing_config.tracer = get_value(Tracer, pattern)
                preprocessing_config.suvr_reference_region = get_value(
                    SUVRReferenceRegions, pattern
                )

            elif isinstance(preprocessing_config, DTIPreprocessingConfig):
                preprocessing_config.dti_measure = get_value(DTIMeasure, pattern)
                preprocessing_config.dti_space = get_value(DTISpace, pattern)

            elif isinstance(preprocessing_config, CustomPreprocessingConfig):
                # TODO: add something to find the custom pattern
                pass
        else:
            raise FileNotFoundError(
                f"The preprocessing folder {preprocessing} does not exist."
            )
        return preprocessing_config

    def get_infos_from_json(
        self, preprocessing_json: Path
    ) -> Tuple[ALL_PREPROCESSING_TYPES, ALL_EXTRACTION_TYPES, Transforms]:
        """Load preprocessing and extraction configuration from JSON file."""
        if not preprocessing_json.is_file():
            raise FileNotFoundError(
                f"The provided preprocessing JSON file {preprocessing_json} does not exist."
            )

        return get_infos_from_json(preprocessing_json)

    def check_test_path(self, test_path: Path, baseline: bool = True) -> Path:
        if baseline:
            train_filename = "train_baseline.tsv"
            label_filename = "labels_baseline.tsv"
        else:
            train_filename = "train.tsv"
            label_filename = "labels.tsv"

        if not (test_path.parent / train_filename).is_file():
            if not (test_path.parent / label_filename).is_file():
                raise ClinicaDLTSVError(
                    f"There is no {train_filename} nor {label_filename} in your folder {test_path.parents[0]} "
                )
            else:
                test_path = test_path.parent / label_filename
        else:
            test_path = test_path.parent / train_filename

        return test_path
