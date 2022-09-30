from typing import List

import click

from clinicadl.utils import cli_param


@click.command(name="get-progression", no_args_is_help=True)
@cli_param.argument.formatted_data_tsv
@click.option(
    "--time_horizon",
    help="Time horizon to analyse stability of the label in the case of a progressive disease.",
    default=36,
    type=int,
)
def cli(formatted_data_tsv, time_horizon):

    from .get_progression import get_progression

    get_progression(formatted_data_tsv, horizon_time=time_horizon)


if __name__ == "__main__":
    cli()
