"""
Launch a random network training.
"""

import argparse
from os import path

from clinicadl.random_search.random_search_utils import random_sampling
from clinicadl.train import train
from clinicadl.utils.maps_manager import check_and_complete, read_json


def launch_search(launch_directory, job_name, options,):

    rs_options = read_json(
        path.join(launch_directory, "random_search.json")
    )
    check_and_complete(rs_options, random_search=True)
    options = random_sampling(rs_options, options)

    options["output_dir"] = path.join(launch_directory, job_name)
    options["model"] = "RandomArchitecture"

    train(options)
