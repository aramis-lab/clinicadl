from typing import Optional, Union

from pydantic import BaseModel, ConfigDict

from clinicadl.caps_dataset.data_config import DataConfig
from clinicadl.caps_dataset.dataloader_config import DataLoaderConfig
from clinicadl.caps_dataset.extraction import config as extraction
from clinicadl.caps_dataset.preprocessing import config as preprocessing
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


def get_preprocessing(preprocessing_type: Preprocessing):
    if preprocessing_type == Preprocessing.T1_LINEAR:
        return preprocessing.T1PreprocessingConfig
    elif preprocessing_type == Preprocessing.PET_LINEAR:
        return preprocessing.PETPreprocessingConfig
    elif preprocessing_type == Preprocessing.FLAIR_LINEAR:
        return preprocessing.FlairPreprocessingConfig
    elif preprocessing_type == Preprocessing.CUSTOM:
        return preprocessing.CustomPreprocessingConfig
    elif preprocessing_type == Preprocessing.DWI_DTI:
        return preprocessing.DTIPreprocessingConfig
    else:
        raise ValueError(
            f"Preprocessing {preprocessing_type.value} is not implemented."
        )


class CapsDatasetConfig(BaseModel):
    """Config class for CapsDataset object.

    caps_directory, preprocessing_json, extract_method, preprocessing
    are arguments that must be passed by the user.

    transforms isn't optional because there is always at least one transform (NanRemoval)
    """

    data: DataConfig
    dataloader: DataLoaderConfig
    extraction: extraction.ExtractionConfig
    preprocessing: preprocessing.PreprocessingConfig
    transforms: TransformsConfig

    # pydantic config
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

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
