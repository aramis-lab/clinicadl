import json
from pathlib import Path
from typing import Any, Dict

import toml

from clinicadl.prepare_data.prepare_data_utils import compute_folder_and_file_type
from clinicadl.utils.preprocessing import path_decoder, path_encoder


def add_default_values(user_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Updates the training parameters defined by the user with the default values in missing fields.

    Args:
        user_dict: dictionary of training parameters defined by the user.

    Returns:
        dictionary of values ready to use for the training process.
    """
    task = user_dict["network_task"]
    # read default values
    clinicadl_root_dir = Path(__file__).parents[2]
    config_path = clinicadl_root_dir / "resources" / "config" / "train_config.toml"
    # from clinicadl.utils.preprocessing import path_decoder
    config_dict = toml.load(config_path)
    # config_dict = path_decoder(config_dict)
    # print(config_dict)

    # task dependent
    config_dict = remove_unused_tasks(config_dict, task)
    # Check that TOML file has the same format as the one in resources
    for section_name in config_dict:
        for key in config_dict[section_name]:
            if key not in user_dict:  # Add value if not present in user_dict
                user_dict[key] = config_dict[section_name][key]

    # Hard-coded options
    if user_dict["n_splits"] and user_dict["n_splits"] > 1:
        user_dict["validation"] = "KFoldSplit"
    else:
        user_dict["validation"] = "SingleSplit"

    user_dict = path_decoder(user_dict)
    return user_dict


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
    from clinicadl.utils.preprocessing import path_decoder

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
            "prepare_dl" "custom_suffix",
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

    if "file_type" not in parameters["preprocessing_dict"]:
        _, file_type = compute_folder_and_file_type(parameters["preprocessing_dict"])
        parameters["preprocessing_dict"]["file_type"] = file_type

    return parameters


def remove_unused_tasks(
    toml_dict: Dict[str, Dict[str, Any]], task: str
) -> Dict[str, Dict[str, Any]]:
    """
    Remove options depending on other tasks than task

    Args:
        toml_dict: dictionary of options as written in a TOML file.
        task: task which will be learnt by the network.

    Returns:
        updated TOML dictionary.
    """
    task_list = ["classification", "regression", "reconstruction"]

    if task not in task_list:
        raise ValueError(
            f"Invalid value for network_task {task}. "
            f"Please task choose in {task_list}."
        )
    task_list.remove(task)

    # Delete all sections related to other tasks
    for other_task in task_list:
        if other_task.capitalize() in toml_dict:
            del toml_dict[other_task.capitalize()]

    return toml_dict
