from typing import Optional, Union

from pydantic import BaseModel, ConfigDict

from clinicadl.caps_dataset.data_config import DataConfig
from clinicadl.caps_dataset.dataloader_config import DataLoaderConfig
from clinicadl.config.config.modality import (
    CustomModalityConfig,
    DTIModalityConfig,
    FlairModalityConfig,
    ModalityConfig,
    PETModalityConfig,
    T1ModalityConfig,
)
from clinicadl.preprocessing import config as preprocessing
from clinicadl.transforms.config import TransformsConfig
from clinicadl.utils.enum import ExtractionMethod, Preprocessing


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
        return T1ModalityConfig
    elif preprocessing == Preprocessing.PET_LINEAR:
        return PETModalityConfig
    elif preprocessing == Preprocessing.FLAIR_LINEAR:
        return FlairModalityConfig
    elif preprocessing == Preprocessing.CUSTOM:
        return CustomModalityConfig
    elif preprocessing == Preprocessing.DWI_DTI:
        return DTIModalityConfig
    else:
        raise ValueError(f"Preprocessing {preprocessing.value} is not implemented.")


class CapsDatasetBase(BaseModel):
    data: DataConfig
    dataloader: DataLoaderConfig
    modality: ModalityConfig
    preprocessing: preprocessing.PreprocessingConfig
    transforms: Optional[TransformsConfig]

    # pydantic config
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)


class CapsDatasetConfig(CapsDatasetBase):
    @classmethod
    def from_preprocessing_and_extraction_method(
        cls,
        preprocessing_type: Union[str, Preprocessing],
        extraction: Union[str, ExtractionMethod],
        **kwargs,
    ):
        return cls(
            data=DataConfig(**kwargs),
            dataloader=DataLoaderConfig(**kwargs),
            modality=get_modality(Preprocessing(preprocessing_type))(**kwargs),
            preprocessing=get_preprocessing(ExtractionMethod(extraction))(**kwargs),
            transforms=TransformsConfig(**kwargs),
        )
