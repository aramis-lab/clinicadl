"""
Launch a random network training.
"""

import argparse
from os import path

from ..tools.deep_learning import read_json
from ..tools.deep_learning.models.random import random_sampling
from .train_multiCNN import train_multi_cnn
from .train_singleCNN import train_single_cnn
from .train_autoencoder import train_autoencoder


def check_and_complete(rs_options):
    """
    This function initializes fields so a random model can be sampled.
    Some fields are mandatory and cannot be initialized by default; this will raise an issue if they are missing.

    Args:
        rs_options: (Namespace) the random search options
    """
    filename = 'random_search.json'

    default_values = {
        "accumulation_steps": 1,
        "baseline": False,
        "channels_limit": 512,
        "d_reduction": "MaxPooling",
        "data_augmentation": False,
        "discarded_slices": 20,
        "dropout": 0,
        "learning_rate": 4,
        "loss": "default",
        "multi_cohort": False,
        "n_conv": 1,
        "network_normalization": "BatchNorm",
        "optimizer": "Adam",
        "unnormalize": False,
        "patch_size": 50,
        "patience": 0,
        "selection_threshold": 0,
        "slice_direction": 0,
        "stride_size": 50,
        "tolerance": 0.0,
        "transfer_learning_path": None,
        "transfer_learning_selection": "best_loss",
        "use_extracted_patches": False,
        "use_extracted_slices": False,
        "wd_bool": True,
        "weight_decay": 4,
        "sampler": "random"
    }
    for name, default_value in default_values.items():
        if not hasattr(rs_options, name):
            setattr(rs_options, name, default_value)

    mandatory_arguments = ['epochs', 'network_type', 'mode',
                           'tsv_path', 'caps_dir', 'diagnoses', 'preprocessing',
                           'n_convblocks', 'first_conv_width', 'n_fcblocks']

    for argument in mandatory_arguments:
        if not hasattr(rs_options, argument):
            raise ValueError(f"The argument {argument} must be specified in {filename}.")


def launch_search(options):

    rs_options = argparse.Namespace()
    rs_options = read_json(rs_options, path.join(options.launch_dir, 'random_search.json'))
    check_and_complete(rs_options)
    random_sampling(rs_options, options)

    options.output_dir = path.join(options.launch_dir, options.name)

    if options.network_type == "autoencoder":
        train_autoencoder(options)
    elif options.network_type == "cnn":
        train_single_cnn(options)
    elif options.network_type == "multicnn":
        train_multi_cnn(options)
