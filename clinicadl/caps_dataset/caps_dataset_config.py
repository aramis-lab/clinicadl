from pathlib import Path
from typing import Optional, Tuple, Union

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
from clinicadl.trainer.trainer_utils import create_parameters_dict, patch_to_read_json
from clinicadl.transforms.config import TransformsConfig
from clinicadl.utils.clinica_utils import (
    FileType,
    bids_nii,
    dwi_dti,
    linear_nii,
    pet_linear_nii,
)
from clinicadl.utils.enum import ExtractionMethod, Preprocessing
from clinicadl.utils.exceptions import MAPSError
from clinicadl.utils.maps_manager_utils import read_json


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
        return T1PreprocessingConfig
    elif preprocessing_type == Preprocessing.PET_LINEAR:
        return PETPreprocessingConfig
    elif preprocessing_type == Preprocessing.FLAIR_LINEAR:
        return FlairPreprocessingConfig
    elif preprocessing_type == Preprocessing.CUSTOM:
        return CustomPreprocessingConfig
    elif preprocessing_type == Preprocessing.DWI_DTI:
        return DTIPreprocessingConfig
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
    preprocessing: PreprocessingConfig
    transforms: TransformsConfig

    # pydantic config
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    @classmethod
    def from_json(cls, config_file: Union[str, Path], maps_path: Union[str, Path]):
        """
        Creates a Trainer from a json configuration file.

        Parameters
        ----------
        config_file : str | Path
            The parameters, stored in a json files.
        maps_path : str | Path
            The folder where the results of a futur training will be stored.

        Returns
        -------
        Trainer
            The Trainer object, instantiated with parameters found in config_file.

        Raises
        ------
        FileNotFoundError
            If config_file doesn't exist.
        """
        config_file = Path(config_file)

        if not (config_file).is_file():
            raise FileNotFoundError(f"No file found at {str(config_file)}.")
        config_dict = patch_to_read_json(read_json(config_file))  # TODO : remove patch

        # read preprocessing for now
        # TODO: remove this
        for key, value in config_dict["preprocessing_dict"]:
            config_dict[key] = value

        preprocessing_type = config_dict["preprocessing"]
        extraction = config_dict["mode"]

        return cls.from_preprocessing_and_extraction_method(
            preprocessing_type=preprocessing_type, extraction=extraction, *config_dict
        )

    @classmethod
    def from_maps(cls, maps_path: Union[str, Path]):
        """
        Creates a CapsDatsetConfig from a MAPS folder.

        Parameters
        ----------
        maps_path : str | Path
            The path of the MAPS folder.

        Returns
        -------
        CapsDatsetConfig
            The config object, instantiated with parameters found in maps_path.

        Raises
        ------
        MAPSError
            If maps_path folder doesn't exist or there is no maps.json file in it.
        """
        maps_path = Path(maps_path)

        if not (maps_path / "maps.json").is_file():
            raise MAPSError(
                f"MAPS was not found at {str(maps_path)}."
                f"To initiate a new MAPS please give a train_dict."
            )
        return cls.from_json(maps_path / "maps.json", maps_path)

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

    def compute_folder_and_file_type(
        self, from_bids: Optional[Path] = None
    ) -> Tuple[str, FileType]:
        preprocessing = self.preprocessing.preprocessing
        if from_bids is not None:
            if isinstance(self.preprocessing, CustomPreprocessingConfig):
                mod_subfolder = Preprocessing.CUSTOM.value
                file_type = FileType(
                    pattern=f"*{self.preprocessing.custom_suffix}",
                    description="Custom suffix",
                )
            else:
                mod_subfolder = preprocessing
                file_type = bids_nii(self.preprocessing)

        elif preprocessing not in Preprocessing:
            raise NotImplementedError(
                f"Extraction of preprocessing {preprocessing} is not implemented from CAPS directory."
            )
        else:
            mod_subfolder = preprocessing.value.replace("-", "_")
            if isinstance(self.preprocessing, T1PreprocessingConfig) or isinstance(
                self.preprocessing, FlairPreprocessingConfig
            ):
                file_type = linear_nii(self.preprocessing)
            elif isinstance(self.preprocessing, PETPreprocessingConfig):
                file_type = pet_linear_nii(self.preprocessing)
            elif isinstance(self.preprocessing, DTIPreprocessingConfig):
                file_type = dwi_dti(self.preprocessing)
            elif isinstance(self.preprocessing, CustomPreprocessingConfig):
                file_type = FileType(
                    pattern=f"*{self.preprocessing.custom_suffix}",
                    description="Custom suffix",
                )
        return mod_subfolder, file_type


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
