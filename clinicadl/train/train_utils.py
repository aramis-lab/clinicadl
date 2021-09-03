import os
from typing import Any, Dict, Optional

import toml


def get_train_dict(
    user_dict: Optional[Dict[str, Dict[str, Any]]], preprocessing_json: str, task: str
) -> Dict[str, Any]:
    """
    Update the user configuration dict with default values of ClinicaDL.

    Args:
        user_dict: user configuration read from TOML file.
        preprocessing_json: path to the JSON file containing preprocessing configuration.
        task: task learnt by the network (example: classification, regression, reconstruction...).
    Returns:
        dictionary of values ready to use for the MapsManager
    """
    # read default values
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(
        *os.path.split(current_file_path)[:-1],
        "resources",
        "config",
        "train_config.toml",
    )
    config_dict = toml.load(config_path)
    # read user specified config and replace default values
    if user_dict is not None:
        for section_name in user_dict:
            if section_name not in config_dict:
                raise IOError(
                    f"{section_name} section is not valid in TOML configuration file. "
                    f"Please see the documentation to see the list of option in TOML configuration file."
                )
            for key in user_dict[section_name]:
                if key not in config_dict[section_name]:
                    raise IOError(
                        f"{key} option in {section_name} is not valid in TOML configuration file. "
                        f"Please see the documentation to see the list of option in TOML configuration file."
                    )
                config_dict[section_name][key] = user_dict[section_name][key]

    train_dict = dict()

    # task dependent
    task_list = ["classification", "regression", "reconstruction"]

    if task not in task_list:
        raise ValueError(
            f"Invalid value for network_task {task}. "
            f"Please task choose in {task_list}."
        )
    task_list.remove(task)

    # Delete all sections related to other tasks
    for other_task in task_list:
        del config_dict[other_task.capitalize()]

    # Standard arguments
    for config_section in config_dict:
        for key in config_dict[config_section]:
            train_dict[key] = config_dict[config_section][key]

    renamed_dict = {
        "normalize": "minmaxnormalization",
        "n_proc": "num_workers",
        "split": "folds",
    }

    for command_name, code_name in renamed_dict.items():
        train_dict[code_name] = train_dict.pop(command_name)

    # GPU exception
    train_dict["use_cpu"] = not train_dict.pop("gpu")

    # Hard-coded optimizer
    train_dict["optimizer"] = "Adam"

    # Mode and preprocessing
    from clinicadl.utils.preprocessing import read_preprocessing

    preprocessing_dict = read_preprocessing(preprocessing_json)
    train_dict["preprocessing_dict"] = preprocessing_dict
    train_dict["mode"] = preprocessing_dict["mode"]

    return train_dict
