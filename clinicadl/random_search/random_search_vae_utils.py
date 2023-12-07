import random
from os import path
from typing import Any, Dict, Tuple

import toml

from clinicadl.train.train_utils import build_train_dict
from clinicadl.utils.exceptions import ClinicaDLConfigurationError
from clinicadl.utils.preprocessing import read_preprocessing

from clinicadl.random_search.random_search_classification_utils import sampling_fn
from clinicadl.random_search.random_search_pythae_parameters import RS_PYTHAE_DICT


def get_vae_space_dict(launch_directory, toml_name):
    """
    Takes a launch directory with a "random_search.toml" file with all the parameters to explore.
    Return a parameters dictionnary randomly sampled
    """

    toml_path = path.join(launch_directory, toml_name)
    if not path.exists(toml_path):
        raise FileNotFoundError(
            f"TOML file {toml} must be written in directory {launch_directory}."
        )

    # load TOML file and create space dict
    toml_options = toml.load(toml_path)
    space_dict = dict()

    # check and read TOML
    if "Random_Search" not in toml_options:
        raise ClinicaDLConfigurationError(
            "Category 'Random_Search' must be defined in the random_search.toml file. "
            "All random search arguments AND options must be defined in this category."
        )

    for key in toml_options["Random_Search"]:
        space_dict[key] = toml_options["Random_Search"][key]

    # Check presence of mandatory arguments
    mandatory_arguments = [
        "network_task",
        "tsv_path",
        "caps_directory",
        "preprocessing_json",
        "first_layer_channels",
        "n_block_encoder",
        "feature_size",
        "latent_space_size",
        "n_block_decoder",
        "last_layer_channels",
        "last_layer_conv",
        "n_layer_per_block_encoder",
        "n_layer_per_block_decoder",
        "block_type",
    ]

    for argument in mandatory_arguments:
        if argument not in space_dict:
            raise ClinicaDLConfigurationError(
                f"The argument {argument} must be specified in the random_search.toml file (Random_Search category)."
            )

    # Make training parameter dict
    train_default = build_train_dict(toml_path, space_dict["network_task"])

    # Mode and preprocessing
    preprocessing_json = path.join(
        space_dict["caps_directory"],
        "tensor_extraction",
        space_dict.pop("preprocessing_json"),
    )

    preprocessing_dict = read_preprocessing(preprocessing_json)
    train_default["preprocessing_dict"] = preprocessing_dict
    train_default["mode"] = preprocessing_dict["mode"]

    # Add the other default parameters to the dictionnary
    # space_dict.update(train_default)
    for k, v in train_default.items():
        if k not in space_dict:
            space_dict[k] = v
    return space_dict


def vae_random_sampling(space_dict):
    # Create parameters dict
    parameters = dict()

    sampling_vae_dict = {
        "accumulation_steps": "fixed",
        "baseline": "fixed",
        "batch_size": "fixed",
        "caps_directory": "fixed",
        "channels_limit": "fixed",
        "compensation": "fixed",
        "data_augmentation": "fixed",
        "deterministic": "fixed",
        "diagnoses": "fixed",
        "epochs": "fixed",
        "evaluation_steps": "fixed",
        "gpu": "fixed",
        "label": "fixed",
        "learning_rate": "fixed",
        "mode": "fixed",
        "multi_cohort": "fixed",
        "n_splits": "fixed",
        "n_proc": "fixed",
        "network_task": "fixed",
        "normalize": "fixed",
        "optimizer": "choice",
        "patience": "fixed",
        "preprocessing_dict": "fixed",
        "sampler": "fixed",
        "seed": "fixed",
        "selection_metrics": "fixed",
        "size_reduction": "fixed",
        "size_reduction_factor": "fixed",
        "split": "fixed",
        "tolerance": "fixed",
        "transfer_path": "fixed",
        "transfer_selection_metric": "fixed",
        "tsv_path": "fixed",
        "wd_bool": "fixed",
        "weight_decay": "fixed",
        "learning_rate": "choice",
        # VAE architecture
        "architecture": "choice",
        "first_layer_channels": "choice",
        "n_block_encoder": "randint",
        "feature_size": "choice",
        "latent_space_size": "choice",
        "n_block_decoder": "randint",
        "last_layer_channels": "choice",
        "last_layer_conv": "choice",
        "n_layer_per_block_encoder": "randint",
        "n_layer_per_block_decoder": "randint",
        "block_type": "choice",
    }
    print(space_dict)
    if space_dict["architecture"] in RS_PYTHAE_DICT.keys():
        sampling_vae_dict.update(RS_PYTHAE_DICT[space_dict["architecture"]])

    for name, sampling_type in sampling_vae_dict.items():
        if name in space_dict:
            sampled_value = sampling_fn(space_dict[name], sampling_type)
            parameters[name] = sampled_value

    return parameters
