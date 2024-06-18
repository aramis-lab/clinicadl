from typing import Optional, Union

from pydantic import BaseModel, ConfigDict

from clinicadl.caps_dataset.data_config import DataConfig
from clinicadl.caps_dataset.dataloader_config import DataLoaderConfig
from clinicadl.caps_dataset.extraction import config as extraction
from clinicadl.caps_dataset.preprocessing.config import (
    CustomPreprocessingConfig,
    DTIPreprocessingConfig,
    FlairPreprocessingConfig,
    PETPreprocessingConfig,
    PreprocessingConfig,
    T1PreprocessingConfig,
)
from clinicadl.transforms.config import TransformsConfig
from clinicadl.utils.enum import ExtractionMethod, Preprocessing


def get_extraction(extract_method: ExtractionMethod):
    if extract_method == ExtractionMethod.ROI:
        return extraction.ExtractionROIConfig
    elif extract_method == ExtractionMethod.SLICE:
        return extraction.ExtractionSliceConfig
    elif extract_method == ExtractionMethod.IMAGE:
        return extraction.ExtractionImageConfig
    elif extract_method == ExtractionMethod.PATCH:
        return extraction.ExtractionPatchConfig
    else:
        raise ValueError(f"Preprocessing {extract_method.value} is not implemented.")


def get_preprocessing(preprocessing: Preprocessing):
    if preprocessing == Preprocessing.T1_LINEAR:
        return T1PreprocessingConfig
    elif preprocessing == Preprocessing.PET_LINEAR:
        return PETPreprocessingConfig
    elif preprocessing == Preprocessing.FLAIR_LINEAR:
        return FlairPreprocessingConfig
    elif preprocessing == Preprocessing.CUSTOM:
        return CustomPreprocessingConfig
    elif preprocessing == Preprocessing.DWI_DTI:
        return DTIPreprocessingConfig
    else:
        raise ValueError(f"Preprocessing {preprocessing.value} is not implemented.")


class CapsDatasetBase(BaseModel):
    data: DataConfig
    dataloader: DataLoaderConfig
    extraction: extraction.ExtractionConfig
    preprocessing: PreprocessingConfig
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
            preprocessing=get_preprocessing(Preprocessing(preprocessing_type))(
                **kwargs
            ),
            extraction=get_extraction(ExtractionMethod(extraction))(**kwargs),
            transforms=TransformsConfig(**kwargs),
        )
