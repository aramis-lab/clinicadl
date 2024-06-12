from typing import Optional, Union

from pydantic import BaseModel, ConfigDict

from clinicadl.caps_dataset.data_config import DataConfig
from clinicadl.caps_dataset.dataloader_config import DataLoaderConfig
from clinicadl.caps_dataset.extraction.config import (
    ExtractionConfig,
    ExtractionImageConfig,
    ExtractionPatchConfig,
    ExtractionROIConfig,
    ExtractionSliceConfig,
)
from clinicadl.caps_dataset.extraction.preprocessing import read_preprocessing
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
        return ExtractionROIConfig
    elif extract_method == ExtractionMethod.SLICE:
        return ExtractionSliceConfig
    elif extract_method == ExtractionMethod.IMAGE:
        return ExtractionImageConfig
    elif extract_method == ExtractionMethod.PATCH:
        return ExtractionPatchConfig
    else:
        raise ValueError(f"Preprocessing {extract_method.value} is not implemented.")


def get_preprocessing(preprocessing: Preprocessing):
    if (
        preprocessing == Preprocessing.T1_EXTENSIVE
        or preprocessing == Preprocessing.T1_LINEAR
    ):
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
    """Config class to specify the CapsDataset.

    caps_directory and preprocessing_json are arguments
    that must be passed by the user.
    """

    data: DataConfig
    dataloader: DataLoaderConfig
    preprocessing: PreprocessingConfig
    extraction: ExtractionConfig
    transforms: Optional[TransformsConfig]

    # pydantic config
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)


class CapsDatasetConfig(CapsDatasetBase):
    @classmethod
    def from_preprocessing_and_extraction_method(
        cls,
        preprocessing_type: Union[str, Preprocessing],
        extraction_type: Union[str, ExtractionMethod],
        **kwargs,
    ):
        return cls(
            data=DataConfig(**kwargs),
            dataloader=DataLoaderConfig(**kwargs),
            preprocessing=get_preprocessing(Preprocessing(preprocessing_type))(
                **kwargs
            ),
            extraction=get_extraction(ExtractionMethod(extraction_type))(**kwargs),
            transforms=TransformsConfig(**kwargs),
        )

    @classmethod
    def from_preprocessing_json(
        cls,
        preprocessing_json: str,
        **kwargs,
    ):
        preprocessing_dict = read_preprocessing(preprocessing_json)
        preprocessing = preprocessing_dict["preprocessing"]
        extraction = preprocessing_dict["mode"]
        kwargs.update(preprocessing_dict)

        return cls(
            data=DataConfig(**kwargs),
            dataloader=DataLoaderConfig(**kwargs),
            preprocessing=get_preprocessing(Preprocessing(preprocessing))(**kwargs),
            extraction=get_extraction(ExtractionMethod(extraction))(**kwargs),
            transforms=TransformsConfig(**kwargs),
        )

    # @computed_field
    # @property
    # def preprocessing_dict(self) -> Dict[str, Any]:
    #     """
    #     Gets the preprocessing dictionary from a preprocessing json file.

    #     Returns
    #     -------
    #     Dict[str, Any]
    #         The preprocessing dictionary.

    #     Raises
    #     ------
    #     ValueError
    #         In case of multi-cohort dataset, if no preprocessing file is found in any CAPS.
    #     """

    #     if (
    #         preprocessing_dict["mode"] == "roi"
    #         and "roi_background_value" not in preprocessing_dict
    #     ):
    #         preprocessing_dict["roi_background_value"] = 0

    #     return preprocessing_dict
