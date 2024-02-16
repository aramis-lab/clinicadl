"""
Launch a random network training.
"""

from pathlib import Path

from clinicadl.random_search.random_search_utils import get_space_dict, random_sampling
from clinicadl.train import train


def launch_search(launch_directory: Path, job_name):
    if not (launch_directory / "random_search.toml").is_file():
        raise FileNotFoundError(
            f"TOML file 'random_search.toml' must be written in directory: {launch_directory}."
        )
    space_options = get_space_dict(launch_directory)
    options = random_sampling(space_options)

    maps_directory = launch_directory / job_name
    split = options.pop("split")
    options["architecture"] = "RandomArchitecture"

    train(maps_directory, options, split)
