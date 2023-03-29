from logging import getLogger
from pathlib import Path

import click

from clinicadl.utils import cli_param


@click.command(name="from_json", no_args_is_help=True)
@click.argument(
    "config_json",
    type=click.Path(exists=True, path_type=Path),
)
@cli_param.argument.output_maps
@click.option(
    "--split",
    "-s",
    type=int,
    # default=(),
    multiple=True,
    help="Train the list of given splits. By default, all the splits are trained.",
)
def cli(
    config_json,
    output_maps_directory,
    split,
):
    """
    Replicate a deep learning training based on a previously created JSON file.
    This is particularly useful to retrain random architectures obtained with a random search.

    CONFIG_JSON is the path to the JSON file with the configuration of the training procedure.

    OUTPUT_MAPS_DIRECTORY is the path to the MAPS folder where outputs and results will be saved.
    """
    from clinicadl.utils.maps_manager.maps_manager_utils import read_json

    from .train import train

    logger = getLogger("clinicadl")
    logger.info(f"Reading JSON file at path {config_json}...")
    train_dict = read_json(config_json)

    train(output_maps_directory, train_dict, split)


if __name__ == "__main__":
    cli()
