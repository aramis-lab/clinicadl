import click

from clinicadl.utils import cli_param


@click.command(name="resume", no_args_is_help=True)
@cli_param.argument.input_maps
def cli(input_maps_directory):
    """Resume training job in specified maps.

    INPUT_MAPS_DIRECTORY is the path to the MAPS folder where training job has started.
    """
    from .resume import automatic_resume

    automatic_resume(input_maps_directory)


if __name__ == "__main__":
    cli()
