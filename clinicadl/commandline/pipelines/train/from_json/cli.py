from logging import getLogger

import click

from clinicadl.commandline import arguments
from clinicadl.commandline.modules_options import (
    cross_validation,
)
from clinicadl.trainer.trainer import Trainer


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

    logger = getLogger("clinicadl")
    logger.info(f"Reading JSON file at path {kwargs['config_file']}...")

    trainer = Trainer.from_json(
        config_file=kwargs["config_file"], maps_path=kwargs["output_maps_directory"]
    )
    trainer.train(split_list=kwargs["split"], overwrite=True)
