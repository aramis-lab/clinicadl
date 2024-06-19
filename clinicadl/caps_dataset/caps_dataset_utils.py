from pathlib import Path
from typing import Dict, Optional, Tuple

from clinicadl.caps_dataset.caps_dataset_config import CapsDatasetConfig
from clinicadl.caps_dataset.preprocessing.config import (
    CustomPreprocessingConfig,
    DTIPreprocessingConfig,
    FlairPreprocessingConfig,
    PETPreprocessingConfig,
    T1PreprocessingConfig,
)
from clinicadl.utils.clinica_utils import (
    FileType,
    bids_nii,
    dwi_dti,
    linear_nii,
    pet_linear_nii,
)
from clinicadl.utils.enum import LinearModality, Preprocessing


def compute_folder_and_file_type(
    config: CapsDatasetConfig, from_bids: Optional[Path] = None
) -> Tuple[str, FileType]:
    preprocessing = config.preprocessing.preprocessing
    if from_bids is not None:
        if isinstance(config.preprocessing, CustomPreprocessingConfig):
            mod_subfolder = Preprocessing.CUSTOM.value
            file_type = FileType(
                pattern=Path(f"*{config.preprocessing.custom_suffix}"),
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
                pattern=Path(f"*{config.preprocessing.custom_suffix}"),
                description="Custom suffix",
            )
    return mod_subfolder, file_type
