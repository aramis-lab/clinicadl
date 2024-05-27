from logging import getLogger
from pathlib import Path

import click

from clinicadl.config import arguments
from clinicadl.config.options import (
    cross_validation,
    reproducibility,
)
from clinicadl.train.tasks import create_training_config


@click.command(name="from_json", no_args_is_help=True)
@arguments.config_file
@arguments.output_maps
@cross_validation.split
def cli(**kwargs):
    """
    Replicate a deep learning training based on a previously created JSON file.
    This is particularly useful to retrain random architectures obtained with a random search.

    CONFIG_JSON is the path to the JSON file with the configuration of the training procedure.

    OUTPUT_MAPS_DIRECTORY is the path to the MAPS folder where outputs and results will be saved.
    """
    from clinicadl.train.trainer import Trainer
    from clinicadl.utils.maps_manager.maps_manager_utils import read_json

    logger = getLogger("clinicadl")
    logger.info(f"Reading JSON file at path {kwargs['config_file']}...")
    config_dict = read_json(kwargs["config_file"])
    # temporary
    config_dict["tsv_directory"] = config_dict["tsv_path"]
    if ("track_exp" in config_dict) and (config_dict["track_exp"] == ""):
        config_dict["track_exp"] = None

    config_dict["maps_dir"] = config_dict["output_maps_directory"]
    config_dict["preprocessing_json"] = None
    ###
    config = create_training_config(config_dict["network_task"])(
        output_maps_directory=kwargs["ouptut_maps"], **config_dict
    )
    trainer = Trainer(config)
    trainer.train(split_list=kwargs["split"], overwrite=True)
