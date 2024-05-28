"""
Launch a random network training.
"""

from pathlib import Path

from clinicadl.train.trainer import Trainer

from .random_search_config import RandomSearchConfig, create_training_config
from .random_search_utils import get_space_dict, random_sampling


def launch_search(launch_directory: Path, job_name):
    if not (launch_directory / "random_search.toml").is_file():
        raise FileNotFoundError(
            f"TOML file 'random_search.toml' must be written in directory: {launch_directory}."
        )
    maps_directory = launch_directory / job_name

    options = get_space_dict(launch_directory)
    # temporary, TODO
    options["tsv_directory"] = options["tsv_path"]
    options["maps_dir"] = maps_directory
    options["preprocessing_json"] = options["preprocessing_dict"]["extract_json"]

    ###

    randomsearch_config = RandomSearchConfig(**options)

    # TODO : modify random_sampling so that it uses randomsearch_config
    # TODO : make something cleaner to merge sampled and fixed parameters
    # TODO : create a RandomSearch object?
    sampled_options = random_sampling(randomsearch_config.model_dump())
    options.update(sampled_options)
    ###
    print(options)
    training_config = create_training_config(options["network_task"])(
        output_maps_directory=maps_directory, **options
    )
    trainer = Trainer(training_config)
    trainer.train(split_list=training_config.cross_validation.split, overwrite=True)
