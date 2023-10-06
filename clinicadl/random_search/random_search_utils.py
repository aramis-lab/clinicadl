import random
from pathlib import Path
from typing import Any, Dict, Tuple

import toml

from clinicadl.train.train_utils import build_train_dict
from clinicadl.utils.exceptions import ClinicaDLConfigurationError
from clinicadl.utils.maps_manager.maps_manager_utils import change_str_to_path
from clinicadl.utils.preprocessing import read_preprocessing


def get_space_dict(launch_directory: Path) -> Dict[str, Any]:
    """Transforms the TOML dictionary in one dimension dictionary."""
    toml_path = launch_directory / "random_search.toml"
    toml_options = toml.load(toml_path)

    if "Random_Search" not in toml_options:
        raise ClinicaDLConfigurationError(
            "Category 'Random_Search' must be defined in the random_search.toml file. "
            "All random search arguments AND options must be defined in this category."
        )

    space_dict = dict()
    for key in toml_options["Random_Search"]:
        space_dict[key] = toml_options["Random_Search"][key]

    space_dict = change_str_to_path(space_dict)
    # Check presence of mandatory arguments
    mandatory_arguments = [
        "network_task",
        "tsv_path",
        "caps_directory",
        "preprocessing_json",
        "n_convblocks",
        "first_conv_width",
        "n_fcblocks",
    ]

    for argument in mandatory_arguments:
        if argument not in space_dict:
            raise ClinicaDLConfigurationError(
                f"The argument {argument} must be specified in the random_search.toml file (Random_Search category)."
            )

    # Default of specific options of random search
    random_search_specific_options = {
        "d_reduction": "MaxPooling",
        "network_normalization": "BatchNorm",
        "channels_limit": 512,
        "n_conv": 1,
        "wd_bool": True,
    }

    for option, value in random_search_specific_options.items():
        if option not in space_dict:
            space_dict[option] = value

    train_default = build_train_dict(toml_path, space_dict["network_task"])

    # Mode and preprocessing
    preprocessing_json = (
        space_dict["caps_directory"]
        / "tensor_extraction"
        / space_dict.pop("preprocessing_json")
    )

    preprocessing_dict = read_preprocessing(preprocessing_json)
    train_default["preprocessing_dict"] = preprocessing_dict
    train_default["mode"] = preprocessing_dict["mode"]

    space_dict.update(train_default)

    return space_dict


def sampling_fn(value, sampling_type: str):
    if isinstance(value, (tuple, list)):
        if sampling_type == "fixed":
            return value
        elif sampling_type == "choice":
            return random.choice(value)
        elif sampling_type == "exponent":
            exponent = random.uniform(*value)
            return 10**-exponent
        elif sampling_type == "randint":
            return random.randint(*value)
        elif sampling_type == "uniform":
            return random.uniform(*value)
        else:
            raise NotImplementedError(
                f"Sampling type {sampling_type} is not implemented"
            )
    else:
        if sampling_type == "exponent":
            return 10**-value
        else:
            return value


def random_sampling(rs_options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Samples all the hyperparameters of the model.
    Args:
        rs_options: parameters of the random search
    Returns:
        options updated to train the model generated randomly
    """

    options = dict()
    sampling_dict = {
        "accumulation_steps": "randint",
        "baseline": "choice",
        "batch_size": "fixed",
        "caps_directory": "fixed",
        "channels_limit": "fixed",
        "compensation": "fixed",
        "data_augmentation": "fixed",
        "deterministic": "fixed",
        "diagnoses": "fixed",
        "dropout": "uniform",
        "epochs": "fixed",
        "evaluation_steps": "fixed",
        "gpu": "fixed",
        "label": "fixed",
        "learning_rate": "exponent",
        "normalize": "choice",
        "mode": "fixed",
        "multi_cohort": "fixed",
        "multi_network": "choice",
        "ssda_netork": "fixed",
        "n_fcblocks": "randint",
        "n_splits": "fixed",
        "n_proc": "fixed",
        "network_task": "fixed",
        "network_normalization": "choice",
        "optimizer": "choice",
        "patience": "fixed",
        "preprocessing_dict": "fixed",
        "sampler": "choice",
        "seed": "fixed",
        "selection_metrics": "fixed",
        "split": "fixed",
        "tolerance": "fixed",
        "transfer_path": "choice",
        "transfer_selection_metric": "choice",
        "tsv_path": "fixed",
        "wd_bool": "choice",
        "weight_decay": "exponent",
    }

    for name, sampling_type in sampling_dict.items():
        if name in rs_options:
            sampled_value = sampling_fn(rs_options[name], sampling_type)
            options[name] = sampled_value

    # Exceptions to classical sampling functions
    if not options["wd_bool"]:
        options["weight_decay"] = 0

    options["convolutions_dict"] = random_conv_sampling(rs_options)

    return options


def random_conv_sampling(rs_options: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Generate random parameters for a random architecture (convolutional part).
    Args:
        rs_options: parameters of the random search
    Returns
        parameters of the convolutions
    """
    n_convblocks = sampling_fn(rs_options["n_convblocks"], "randint")
    first_conv_width = sampling_fn(rs_options["first_conv_width"], "choice")
    d_reduction = sampling_fn(rs_options["d_reduction"], "choice")

    # Sampling the parameters of each convolutional block
    convolutions = dict()
    current_in_channels = None
    current_out_channels = first_conv_width
    for i in range(n_convblocks):
        conv_dict = dict()
        conv_dict["in_channels"] = current_in_channels
        conv_dict["out_channels"] = current_out_channels

        current_in_channels, current_out_channels = update_channels(
            current_out_channels, rs_options["channels_limit"]
        )
        conv_dict["n_conv"] = sampling_fn(rs_options["n_conv"], "choice")
        conv_dict["d_reduction"] = d_reduction
        convolutions["conv" + str(i)] = conv_dict

    return convolutions


def update_channels(out_channels: int, channels_limit: int = 512) -> Tuple[int, int]:
    in_channels = out_channels
    if out_channels < channels_limit:
        out_channels = 2 * out_channels

    return in_channels, out_channels
