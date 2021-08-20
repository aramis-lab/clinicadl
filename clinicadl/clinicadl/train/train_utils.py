import os

import toml


def get_train_dict(configuration_toml, preprocessing_json, task):
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
    if configuration_toml is not None:
        user_dict = toml.load(configuration_toml)
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
        del config_dict[other_task.upper()]

    # Standard arguments
    for config_section in config_dict:
        for key in config_dict[config_section]:
            train_dict[key] = config_dict[config_section]

    renamed_dict = {
        "normalize": "minmaxnormalization",
        "n_proc": "num_workers",
        "split": "folds",
    }

    for command_name, code_name in renamed_dict:
        train_dict[code_name] = train_dict.pop(command_name)

    # GPU exception
    train_dict["use_cpu"] = not train_dict.pop("gpu")

    # Hard-coded optimizer
    train_dict["optimizer"] = "Adam"

    # Mode and preprocessing
    from clinicadl.utils.preprocessing import read_preprocessing

    preprocessing_dict = read_preprocessing(preprocessing_json)
    train_dict.update(preprocessing_dict)

    return train_dict
