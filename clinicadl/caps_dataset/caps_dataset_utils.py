from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from clinicadl.caps_dataset.caps_dataset_config import CapsDatasetConfig
from clinicadl.caps_dataset.preprocessing.config import (
    CustomPreprocessingConfig,
    DTIPreprocessingConfig,
    FlairPreprocessingConfig,
    PETPreprocessingConfig,
    T1PreprocessingConfig,
)
from clinicadl.caps_dataset.preprocessing.utils import (
    bids_nii,
    dwi_dti,
    linear_nii,
    pet_linear_nii,
)
from clinicadl.utils.clinica_utils import FileType
from clinicadl.utils.enum import Preprocessing
from clinicadl.utils.exceptions import ClinicaDLArgumentError


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


def find_file_type(config: CapsDatasetConfig) -> FileType:
    if isinstance(config.preprocessing, T1PreprocessingConfig):
        file_type = linear_nii(config.preprocessing)
    elif isinstance(config.preprocessing, PETPreprocessingConfig):
        if (
            config.preprocessing.tracer is None
            or config.preprocessing.suvr_reference_region is None
        ):
            raise ClinicaDLArgumentError(
                "`tracer` and `suvr_reference_region` must be defined "
                "when using `pet-linear` preprocessing."
            )
        file_type = pet_linear_nii(config.preprocessing)
    else:
        raise NotImplementedError(
            f"Generation of synthetic data is not implemented for preprocessing {config.preprocessing.preprocessing.value}"
        )

    return file_type


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
    from clinicadl.utils.iotools.utils import path_decoder

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

    from clinicadl.caps_dataset.caps_dataset_config import CapsDatasetConfig

    config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction=parameters["mode"],
        preprocessing_type=parameters["preprocessing"],
        **parameters,
    )
    if "file_type" not in parameters["preprocessing_dict"]:
        _, file_type = compute_folder_and_file_type(config)
        parameters["preprocessing_dict"]["file_type"] = file_type.model_dump()

    return parameters
