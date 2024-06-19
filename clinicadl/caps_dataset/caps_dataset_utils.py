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
from clinicadl.utils.enum import LinearModality, Preprocessing


def compute_folder_and_file_type(
    config: CapsDatasetConfig, from_bids: Optional[Path] = None
) -> Tuple[str, Dict[str, str]]:
    from clinicadl.utils.clinica_utils import (
        bids_nii,
        dwi_dti,
        linear_nii,
        pet_linear_nii,
    )

    preprocessing = config.preprocessing.preprocessing
    if from_bids is not None:
        if isinstance(config.preprocessing, CustomPreprocessingConfig):
            mod_subfolder = Preprocessing.CUSTOM.value
            file_type = {
                "pattern": f"*{config.preprocessing.custom_suffix}",
                "description": "Custom suffix",
            }
        else:
            mod_subfolder = preprocessing
            file_type = bids_nii(preprocessing)

    elif preprocessing not in Preprocessing:
        raise NotImplementedError(
            f"Extraction of preprocessing {preprocessing} is not implemented from CAPS directory."
        )
    else:
        mod_subfolder = preprocessing.value.replace("-", "_")
        if isinstance(config.preprocessing, T1PreprocessingConfig):
            file_type = linear_nii(
                LinearModality.T1W, config.extraction.use_uncropped_image
            )

        elif isinstance(config.preprocessing, FlairPreprocessingConfig):
            file_type = linear_nii(
                LinearModality.FLAIR, config.extraction.use_uncropped_image
            )

        elif isinstance(config.preprocessing, PETPreprocessingConfig):
            file_type = pet_linear_nii(
                config.preprocessing.tracer,
                config.preprocessing.suvr_reference_region,
                config.extraction.use_uncropped_image,
            )
        elif isinstance(config.preprocessing, DTIPreprocessingConfig):
            file_type = dwi_dti(
                config.preprocessing.dti_measure,
                config.preprocessing.dti_space,
            )
        elif isinstance(config.preprocessing, CustomPreprocessingConfig):
            file_type = {
                "pattern": f"*{config.preprocessing.custom_suffix}",
                "description": "Custom suffix",
            }
            # custom_suffix["use_uncropped_image"] = None

    return mod_subfolder, file_type
