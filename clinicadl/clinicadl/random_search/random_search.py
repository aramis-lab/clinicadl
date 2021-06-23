"""
Launch a random network training.
"""

import argparse
from os import path

from clinicadl.random_search.random_search_utils import random_sampling
from clinicadl.tools.deep_learning import check_and_complete, read_json
from clinicadl.train.train_autoencoder import train_autoencoder
from clinicadl.train.train_multiCNN import train_multi_cnn
from clinicadl.train.train_singleCNN import train_single_cnn


def launch_search(options):

    rs_options = argparse.Namespace()
    rs_options = read_json(
        rs_options, path.join(options.launch_dir, "random_search.json")
    )
    check_and_complete(rs_options, random_search=True)
    random_sampling(rs_options, options)

    options.output_dir = path.join(options.launch_dir, options.name)

    if options.network_type == "autoencoder":
        train_autoencoder(options)
    elif options.network_type == "cnn":
        train_single_cnn(options)
    elif options.network_type == "multicnn":
        train_multi_cnn(options)
