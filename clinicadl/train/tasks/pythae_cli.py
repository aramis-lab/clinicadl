import os

import click

from clinicadl import MapsManager
from clinicadl.train.train_utils import build_train_dict
from clinicadl.utils.cli_param import train_option
from clinicadl.utils.preprocessing import read_preprocessing


@click.command(name="pythae", no_args_is_help=True)
# Mandatory arguments
@train_option.caps_directory
@train_option.preprocessing_json
@train_option.tsv_directory
@train_option.output_maps
# Options
@train_option.config_file
# Cross validation
# @train_option.n_splits
@train_option.split
def cli(
    caps_directory,
    preprocessing_json,
    tsv_directory,
    output_maps_directory,
    config_file,
    split,
):
    """
    Train a deep learning model to learn a vae variant using Pythae on neuroimaging data.

    CAPS_DIRECTORY is the CAPS folder from where tensors will be loaded.

    PREPROCESSING_JSON is the name of the JSON file in CAPS_DIRECTORY/tensor_extraction folder where
    all information about extraction are stored in order to read the wanted tensors.

    TSV_DIRECTORY is a folder were TSV files defining train and validation sets are stored.

    OUTPUT_MAPS is the path to the MAPS folder where outputs and results will be saved.

    Options for this command can be input by providing a configuration file in TOML format.
    For more details, please visit the documentation:
    https://clinicadl.readthedocs.io/en/stable/Train/Introduction/#configuration-file
    """

    parameters = build_train_dict(config_file, "reconstruction")

    parameters["network_task"] = "reconstruction"
    parameters["caps_directory"] = caps_directory
    parameters["tsv_path"] = tsv_directory
    parameters["mode"] = "image"
    # parameters["input_size"] = (1, 80, 96, 80)
    preprocessing_json_path = os.path.join(
        caps_directory,
        "tensor_extraction",
        preprocessing_json,
    )
    parameters["preprocessing_dict"] = read_preprocessing(preprocessing_json_path)

    maps_manager = MapsManager(output_maps_directory, parameters, verbose="info")
    # launch training procedure for Pythae
    maps_manager.train_pythae(split_list=split)
