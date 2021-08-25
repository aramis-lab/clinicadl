"""
Launch a random network training.
"""

from os import path

import toml

from clinicadl.random_search.random_search_utils import get_space_dict, random_sampling
from clinicadl.train import train


def launch_search(launch_directory, job_name, options):

    if not path.exists(path.join(launch_directory, "random_search.toml")):
        raise ValueError(
            f"TOML file 'random_search' must be written in directory {launch_directory}."
        )
    toml_options = toml.load(path.join(launch_directory, "random_search.toml"))
    space_options = get_space_dict(toml_options)
    options = random_sampling(space_options, options)

    maps_directory = path.join(launch_directory, job_name)
    folds = options.pop("folds")
    options["architecture"] = "RandomArchitecture"

    train(maps_directory, options, folds)
