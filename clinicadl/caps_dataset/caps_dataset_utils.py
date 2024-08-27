from pathlib import Path
from typing import Optional, Tuple

from clinicadl.caps_dataset.caps_dataset_config import CapsDatasetConfig
from clinicadl.caps_dataset.preprocessing.config import (
    CustomPreprocessingConfig,
    DTIPreprocessingConfig,
    FlairPreprocessingConfig,
    PETPreprocessingConfig,
    T1PreprocessingConfig,
)
from clinicadl.caps_dataset.preprocessing.utils import (
    bids_nii,
    dwi_dti,
    linear_nii,
    pet_linear_nii,
)
from clinicadl.utils.enum import Preprocessing
from clinicadl.utils.exceptions import ClinicaDLArgumentError
from clinicadl.utils.iotools.clinica_utils import FileType


def compute_folder_and_file_type(
    config: CapsDatasetConfig, from_bids: Optional[Path] = None
) -> Tuple[str, FileType]:
    preprocessing = config.preprocessing.preprocessing
    if from_bids is not None:
        if isinstance(config.preprocessing, CustomPreprocessingConfig):
            mod_subfolder = Preprocessing.CUSTOM.value
            file_type = FileType(
                pattern=f"*{config.preprocessing.custom_suffix}",
                description="Custom suffix",
            )
        else:
            mod_subfolder = preprocessing
            file_type = bids_nii(config.preprocessing)

    elif preprocessing not in Preprocessing:
        raise NotImplementedError(
            f"Extraction of preprocessing {preprocessing} is not implemented from CAPS directory."
        )
    else:
        mod_subfolder = preprocessing.value.replace("-", "_")
        if isinstance(config.preprocessing, T1PreprocessingConfig) or isinstance(
            config.preprocessing, FlairPreprocessingConfig
        ):
            file_type = linear_nii(config.preprocessing)
        elif isinstance(config.preprocessing, PETPreprocessingConfig):
            file_type = pet_linear_nii(config.preprocessing)
        elif isinstance(config.preprocessing, DTIPreprocessingConfig):
            file_type = dwi_dti(config.preprocessing)
        elif isinstance(config.preprocessing, CustomPreprocessingConfig):
            file_type = FileType(
                pattern=f"*{config.preprocessing.custom_suffix}",
                description="Custom suffix",
            )
    return mod_subfolder, file_type


def find_file_type(config: CapsDatasetConfig) -> FileType:
    if isinstance(config.preprocessing, T1PreprocessingConfig):
        file_type = linear_nii(config.preprocessing)
    elif isinstance(config.preprocessing, PETPreprocessingConfig):
        if (
            config.preprocessing.tracer is None
            or config.preprocessing.suvr_reference_region is None
        ):
            raise ClinicaDLArgumentError(
                "`tracer` and `suvr_reference_region` must be defined "
                "when using `pet-linear` preprocessing."
            )
        file_type = pet_linear_nii(config.preprocessing)
    else:
        raise NotImplementedError(
            f"Generation of synthetic data is not implemented for preprocessing {config.preprocessing.preprocessing.value}"
        )

    return file_type
