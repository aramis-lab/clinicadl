"""
Launch a random network training.
"""

import argparse
from os import path

from clinicadl.random_search.random_search_utils import random_sampling
from clinicadl.train import train
from clinicadl.utils.maps_manager import check_and_complete, read_json


def launch_search(launch_directory, job_name, options,):

    # TODO: Change default values according to config TOML
    rs_options = read_json(
        json_path=path.join(launch_directory, "random_search.json")
    )
    check_and_complete(rs_options, random_search=True)
    options = random_sampling(rs_options, options)

    maps_directory = path.join(launch_directory, job_name)
    folds = options.pop("split")
    options["architecture"] = "RandomArchitecture"

    train(maps_directory, options, folds)
