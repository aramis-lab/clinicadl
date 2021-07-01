"""
Launch a random network training.
"""

import argparse
from os import path

from clinicadl.random_search.random_search_utils import random_sampling
from clinicadl.train.train_multiCNN import train_multi_cnn
from clinicadl.train.train_singleCNN import train_single_cnn
from clinicadl.utils.maps_manager import check_and_complete, read_json


def launch_search(options):

    rs_options = argparse.Namespace()
    rs_options = read_json(
        rs_options, path.join(options.launch_dir, "random_search.json")
    )
    check_and_complete(rs_options, random_search=True)
    random_sampling(rs_options, options)

    options.output_dir = path.join(options.launch_dir, options.name)
    options.model = "RandomArchitecture"

    if options.network_type == "autoencoder":
        raise NotImplementedError(
            "Random architectures for autoencoders were not implemented."
        )
    elif options.network_type == "cnn":
        train_single_cnn(options)
    elif options.network_type == "multicnn":
        train_multi_cnn(options)
