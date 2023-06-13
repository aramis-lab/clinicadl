"""
Launch a random network training.
"""

from os import path


def launch_search(launch_directory, job_name):
    from clinicadl.random_search.random_search_classification_utils import (
        classification_random_sampling,
        get_classification_space_dict,
    )
    from clinicadl.train import train

    if not path.exists(path.join(launch_directory, "random_search.toml")):
        raise FileNotFoundError(
            f"TOML file 'random_search.toml' must be written in directory {launch_directory}."
        )
    space_options = get_classification_space_dict(launch_directory)
    options = classification_random_sampling(space_options)

    maps_directory = path.join(launch_directory, job_name)
    split = options.pop("split")
    options["architecture"] = "RandomArchitecture"

    train(maps_directory, options, split)


def launch_vae_search(launch_directory, job_name):
    from clinicadl.random_search.random_search_vae_utils import (
        get_vae_space_dict,
        vae_random_sampling,
    )
    from clinicadl.utils.maps_manager import MapsManager

    space_options = get_vae_space_dict(launch_directory)
    parameters = vae_random_sampling(space_options)
    parameters["architecture"] = "pythae_VAE"
    # initialise maps
    maps_dir = path.join(launch_directory, job_name)
    maps_manager = MapsManager(maps_dir, parameters, verbose="info")
    # launch training procedure for Pythae
    maps_manager.train_pythae()
