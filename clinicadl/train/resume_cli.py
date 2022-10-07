import click

from clinicadl.utils import cli_param
from clinicadl.utils.cli_param import train_option


@click.command(name="resume", no_args_is_help=True)
@cli_param.argument.input_maps
@train_option.split
def cli(input_maps_directory, split):
    """Resume training job in specified maps.

    INPUT_MAPS_DIRECTORY is the path to the MAPS folder where training job has started.
    """
    from .resume import automatic_resume

    automatic_resume(input_maps_directory, user_split_list=split)


if __name__ == "__main__":
    cli()
