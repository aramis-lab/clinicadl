import click

from clinicadl.commandline import arguments
from clinicadl.commandline.modules_options import (
    split,
)
from clinicadl.trainer.trainer import Trainer


@click.command(name="resume", no_args_is_help=True)
@arguments.input_maps
@split.split
def cli(input_maps_directory, split):
    """Resume training job in specified maps.

    INPUT_MAPS_DIRECTORY is the path to the MAPS folder where training job has started.
    """
    trainer = Trainer.from_maps(input_maps_directory)
    trainer.resume()
