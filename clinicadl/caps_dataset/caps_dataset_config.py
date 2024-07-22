import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
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
from clinicadl.caps_dataset.preprocessing.utils import (
    bids_nii,
    dwi_dti,
    linear_nii,
    pet_linear_nii,
)
from clinicadl.transforms.config import TransformsConfig
from clinicadl.utils.enum import ExtractionMethod, Preprocessing
from clinicadl.utils.exceptions import ClinicaDLArgumentError, MAPSError
from clinicadl.utils.iotools.clinica_utils import FileType
from clinicadl.utils.iotools.trainer_utils import patch_to_read_json
from clinicadl.utils.iotools.utils import path_decoder


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
    def from_data_group(
        cls,
        maps_path: Union[str, Path],
        data_group: str,
        caps_directory: Optional[Path] = None,
        data_tsv: Optional[Path] = None,
        overwrite: bool = False,
    ):
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
        maps_path = Path(maps_path)
        group_dir = maps_path / "groups" / data_group

        if not group_dir.is_dir() and caps_directory is None:
            raise ClinicaDLArgumentError(
                f"The data group {data_group} does not already exist. "
                f"Please specify a caps_directory and a tsv_path to create this data group."
            )
        elif group_dir.is_dir() and overwrite and data_group in ["train", "validation"]:
            raise MAPSError("Cannot overwrite train or validation data group.")

        elif group_dir.is_dir() and not overwrite:
            raise ClinicaDLArgumentError(
                f"Data group {data_group} is already defined. "
                f"Please do not give any caps_directory, tsv_path or multi_cohort to use it. "
                f"To erase {data_group} please set overwrite to True."
            )

        config = cls.from_json(config_file=maps_path / "maps.json")

        if group_dir.is_dir() and caps_directory is None:
            config.data.caps_directory = read_json(group_dir / "maps.json")[
                "caps_directory"
            ]
            config.data.data_tsv = group_dir / "data.tsv"

        elif not group_dir.is_dir() and caps_directory is not None:
            config.data.caps_directory = caps_directory
            config.data.data_tsv = data_tsv

        config.data.data_df = pd.read_csv(config.data.data_tsv, sep="\t")

        return config

    @classmethod
    def from_json(cls, config_file: Union[str, Path]):
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


def read_json(json_path: Path) -> Dict[str, Any]:
    """
    Ensures retro-compatibility between the different versions of ClinicaDL.

    Parameters
    ----------
    json_path: Path
        path to the JSON file summing the parameters of a MAPS.

    Returns
    -------
    A dictionary of training parameters.
    """

    with json_path.open(mode="r") as f:
        parameters = json.load(f, object_hook=path_decoder)
    # Types of retro-compatibility
    # Change arg name: ex network --> model
    # Change arg value: ex for preprocessing: mni --> t1-extensive
    # New arg with default hard-coded value --> discarded_slice --> 20
    retro_change_name = {
        "model": "architecture",
        "multi": "multi_network",
        "minmaxnormalization": "normalize",
        "num_workers": "n_proc",
        "mode": "extract_method",
    }

    retro_add = {
        "optimizer": "Adam",
        "loss": None,
    }

    for old_name, new_name in retro_change_name.items():
        if old_name in parameters:
            parameters[new_name] = parameters[old_name]
            del parameters[old_name]

    for name, value in retro_add.items():
        if name not in parameters:
            parameters[name] = value

    if "extract_method" in parameters:
        parameters["mode"] = parameters["extract_method"]
    # Value changes
    if "use_cpu" in parameters:
        parameters["gpu"] = not parameters["use_cpu"]
        del parameters["use_cpu"]
    if "nondeterministic" in parameters:
        parameters["deterministic"] = not parameters["nondeterministic"]
        del parameters["nondeterministic"]

    # Build preprocessing_dict
    if "preprocessing_dict" not in parameters:
        parameters["preprocessing_dict"] = {"mode": parameters["mode"]}
        preprocessing_options = [
            "preprocessing",
            "use_uncropped_image",
            "prepare_dl",
            "custom_suffix",
            "tracer",
            "suvr_reference_region",
            "patch_size",
            "stride_size",
            "slice_direction",
            "slice_mode",
            "discarded_slices",
            "roi_list",
            "uncropped_roi",
            "roi_custom_suffix",
            "roi_custom_template",
            "roi_custom_mask_pattern",
        ]
        for preprocessing_var in preprocessing_options:
            if preprocessing_var in parameters:
                parameters["preprocessing_dict"][preprocessing_var] = parameters[
                    preprocessing_var
                ]
                del parameters[preprocessing_var]

    # Add missing parameters in previous version of extract
    if "use_uncropped_image" not in parameters["preprocessing_dict"]:
        parameters["preprocessing_dict"]["use_uncropped_image"] = False

    if (
        "prepare_dl" not in parameters["preprocessing_dict"]
        and parameters["mode"] != "image"
    ):
        parameters["preprocessing_dict"]["prepare_dl"] = False

    if (
        parameters["mode"] == "slice"
        and "slice_mode" not in parameters["preprocessing_dict"]
    ):
        parameters["preprocessing_dict"]["slice_mode"] = "rgb"

    if "preprocessing" not in parameters:
        parameters["preprocessing"] = parameters["preprocessing_dict"]["preprocessing"]

    config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction=parameters["mode"],
        preprocessing_type=parameters["preprocessing"],
        **parameters,
    )
    if "file_type" not in parameters["preprocessing_dict"]:
        _, file_type = compute_folder_and_file_type(config)
        parameters["preprocessing_dict"]["file_type"] = file_type.model_dump()

    return parameters
