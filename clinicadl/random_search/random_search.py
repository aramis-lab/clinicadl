"""
Launch a random network training.
"""

from os import path

import toml

from clinicadl.random_search.random_search_utils import get_space_dict, random_sampling
from clinicadl.train import train


def launch_search(launch_directory, job_name):

    if not path.exists(path.join(launch_directory, "random_search.toml")):
        raise ValueError(
            f"TOML file 'random_search' must be written in directory {launch_directory}."
        )
    space_options = get_space_dict(launch_directory)
    options = random_sampling(space_options)

    maps_directory = path.join(launch_directory, job_name)
    split = options.pop("split")
    options["architecture"] = "RandomArchitecture"

    train(maps_directory, options, split)
