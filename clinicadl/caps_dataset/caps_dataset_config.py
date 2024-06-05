import abc
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from pydantic import BaseModel, computed_field

from clinicadl.caps_dataset.data_config import ConfigDict, DataConfig
from clinicadl.caps_dataset.data_utils import check_multi_cohort_tsv
from clinicadl.caps_dataset.dataloader_config import DataLoaderConfig
from clinicadl.config.config import modality
from clinicadl.generate import generate_config as generate_type
from clinicadl.generate.generate_config import GenerateConfig
from clinicadl.preprocessing import config as preprocessing
from clinicadl.utils.enum import ExtractionMethod, GenerateType, Preprocessing
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLTSVError,
    DownloadError,
)


def get_preprocessing(extract_method: ExtractionMethod):
    if extract_method == ExtractionMethod.ROI:
        return preprocessing.PreprocessingROIConfig
    elif extract_method == ExtractionMethod.SLICE:
        return preprocessing.PreprocessingSliceConfig
    elif extract_method == ExtractionMethod.IMAGE:
        return preprocessing.PreprocessingImageConfig
    elif extract_method == ExtractionMethod.PATCH:
        return preprocessing.PreprocessingPatchConfig
    else:
        raise ValueError(f"Modality {extract_method.value} is not implemented.")


def get_modality(preprocessing: Preprocessing):
    if (
        preprocessing == Preprocessing.T1_EXTENSIVE
        or preprocessing == Preprocessing.T1_LINEAR
    ):
        return modality.T1ModalityConfig
    elif preprocessing == Preprocessing.PET_LINEAR:
        return modality.PETModalityConfig
    elif preprocessing == Preprocessing.FLAIR_LINEAR:
        return modality.FlairModalityConfig
    elif preprocessing == Preprocessing.CUSTOM:
        return modality.CustomModalityConfig
    elif preprocessing == Preprocessing.DWI_DTI:
        return modality.DTIModalityConfig
    else:
        raise ValueError(f"Preprocessing {preprocessing.value} is not implemented.")


def get_generate(generate: Union[str, GenerateType]):
    generate = GenerateType(generate)
    if generate == GenerateType.ART:
        return generate_type.GenerateArtifactsConfig
    elif generate == GenerateType.RAN:
        return generate_type.GenerateRandomConfig
    elif generate == GenerateType.SHE:
        return generate_type.GenerateSheppLoganConfig
    elif generate == GenerateType.HYP:
        return generate_type.GenerateHypometabolicConfig
    elif generate == GenerateType.TRI:
        return generate_type.GenerateTrivialConfig
    else:
        raise ValueError(f"GenerateType {generate.value} is not available.")


class CapsDatasetBase(BaseModel):
    data: DataConfig
    modality: modality.ModalityConfig
    preprocessing: preprocessing.PreprocessingConfig

    # pydantic config
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)


def create_caps_dataset_config(
    preprocessing: Union[str, Preprocessing], extract: Union[str, ExtractionMethod]
):
    try:
        preprocessing_type = Preprocessing(preprocessing)
    except ClinicaDLArgumentError:
        print("Invalid preprocessing configuration")

    try:
        extract_method = ExtractionMethod(extract)
    except ClinicaDLArgumentError:
        print("Invalid preprocessing configuration")

    class CapsDatasetConfig(CapsDatasetBase):
        modality: get_modality(preprocessing_type)
        preprocessing: get_preprocessing(extract_method)

        def __init__(self, **kwargs):
            super().__init__(data=kwargs, modality=kwargs, preprocessing=kwargs)

    return CapsDatasetConfig
