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

    # Fill train_dict
    # for config_section in config_dict:
    #     for key in config_section:
    #         train_dict[key] = config_dict[config_section]

    # From config file
    train_dict = {
        "accumulation_steps": config_dict["Optimization"]["accumulation_steps"],
        "architecture": config_dict["Model"]["architecture"],
        "baseline": config_dict["Data"]["baseline"],
        "batch_size": config_dict["Computational"]["batch_size"],
        "compensation": config_dict["Reproducibility"]["compensation"],
        "data_augmentation": config_dict["Data"]["data_augmentation"],
        "deterministic": config_dict["Reproducibility"]["deterministic"],
        "diagnoses": config_dict["Data"]["diagnoses"],
        "dropout": config_dict["Architecture"]["dropout"],
        "epochs": config_dict["Optimization"]["epochs"],
        "evaluation_steps": config_dict["Computational"]["evaluation_steps"],
        "learning_rate": config_dict["Optimization"]["learning_rate"],
        "minmaxnormalization": config_dict["Data"]["normalize"],
        "multi_network": config_dict["Model"]["multi_network"],
        "multi_cohort": config_dict["Data"]["multi_cohort"],
        "n_splits": config_dict["Cross_validation"]["n_splits"],
        "num_workers": config_dict["Computational"]["n_proc"],
        "patience": config_dict["Optimization"]["patience"],
        "folds": config_dict["Cross_validation"]["split"],
        "seed": config_dict["Reproducibility"]["seed"],
        "tolerance": config_dict["Optimization"]["tolerance"],
        "transfer_path": config_dict["Transfer_learning"]["transfer_path"],
        "transfer_learning_selection": config_dict["Transfer_learning"][
            "transfer_selection_metric"
        ],
        "use_cpu": not config_dict["Computational"]["use_gpu"],
        "weight_decay": config_dict["Optimization"]["weight_decay"],
        "sampler": config_dict["Data"]["sampler"],
    }

    # task dependent
    if task == "classification":
        train_dict["loss"] = config_dict["Classification"]["optimization_metric"]
        train_dict["selection_metrics"] = config_dict["Classification"][
            "selection_metrics"
        ]
        train_dict["label"] = config_dict["Classification"]["label"]
    elif task == "regression":
        train_dict["loss"] = config_dict["Regression"]["optimization_metric"]
        train_dict["selection_metrics"] = config_dict["Regression"]["selection_metrics"]
        train_dict["label"] = config_dict["Regression"]["label"]
    elif task == "reconstruction":
        train_dict["loss"] = config_dict["Reconstruction"]["optimization_metric"]
        train_dict["selection_metrics"] = config_dict["Reconstruction"][
            "selection_metrics"
        ]
    else:
        raise ValueError("Invalid network_task")

    # optimizer
    train_dict["optimizer"] = "Adam"

    # use extracted features
    train_dict["use_extracted_features"] = config_dict["Mode"]["use_extracted_features"]

    # Mode and preprocessing
    from clinicadl.utils.preprocessing import read_preprocessing

    preprocessing_dict = read_preprocessing(preprocessing_json)
    train_dict.update(preprocessing_dict)

    return train_dict
