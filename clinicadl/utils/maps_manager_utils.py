import json
from pathlib import Path
from typing import Any, Dict

import toml

# from clinicadl.caps_dataset.caps_dataset_config import compute_folder_and_file_type
from clinicadl.utils.iotools.utils import path_decoder


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
