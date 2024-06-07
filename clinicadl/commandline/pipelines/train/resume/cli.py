import click

from clinicadl.commandline import arguments
from clinicadl.commandline.modules_options import (
    cross_validation,
)


@click.command(name="resume", no_args_is_help=True)
@arguments.input_maps
@cross_validation.split
def cli(input_maps_directory, split):
    """Resume training job in specified maps.

    INPUT_MAPS_DIRECTORY is the path to the MAPS folder where training job has started.
    """
    from clinicadl.train.resume import automatic_resume

    automatic_resume(input_maps_directory, user_split_list=split)
