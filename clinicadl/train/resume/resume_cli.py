import click

from clinicadl.utils import cli_param


@click.command(name="resume", no_args_is_help=True)
@cli_param.argument.input_maps
@cli_param.option_group.cross_validation.option(
    "--split",
    "-s",
    type=int,
    multiple=True,
    help="Train the list of given splits. By default, all the splits are trained.",
)
def cli(input_maps_directory, split):
    """Resume training job in specified maps.

    INPUT_MAPS_DIRECTORY is the path to the MAPS folder where training job has started.
    """
    from .resume import automatic_resume

    automatic_resume(input_maps_directory, user_split_list=split)
