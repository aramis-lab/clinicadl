from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict

from clinicadl.caps_dataset.caps_dataset_config import CapsDatasetConfig
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

    preprocessing = Preprocessing(
        config.preprocessing.preprocessing
    )  # replace("-", "_")
    if from_bids is not None:
        if preprocessing == Preprocessing.CUSTOM:
            mod_subfolder = Preprocessing.CUSTOM.value
            file_type = {
                "pattern": f"*{config.modality.custom_suffix}",
                "description": "Custom suffix",
            }
        else:
            mod_subfolder = preprocessing
            file_type = bids_nii(preprocessing)

    elif preprocessing not in Preprocessing:
        raise NotImplementedError(
            f"Extraction of preprocessing {config.preprocessing.preprocessing.value} is not implemented from CAPS directory."
        )
    else:
        mod_subfolder = preprocessing.value.replace("-", "_")
        if preprocessing == Preprocessing.T1_LINEAR:
            file_type = linear_nii(
                LinearModality.T1W, config.preprocessing.use_uncropped_image
            )

        elif preprocessing == Preprocessing.FLAIR_LINEAR:
            file_type = linear_nii(
                LinearModality.FLAIR, config.preprocessing.use_uncropped_image
            )

        elif preprocessing == Preprocessing.PET_LINEAR:
            file_type = pet_linear_nii(
                config.modality.tracer,
                config.modality.suvr_reference_region,
                config.preprocessing.use_uncropped_image,
            )
        elif preprocessing == Preprocessing.DWI_DTI:
            file_type = dwi_dti(
                config.modality.dti_measure,
                config.modality.dti_space,
            )
        elif preprocessing == Preprocessing.CUSTOM:
            file_type = {
                "pattern": f"*{config.modality.custom_suffix}",
                "description": "Custom suffix",
            }
            # custom_suffix["use_uncropped_image"] = None

    return mod_subfolder, file_type
