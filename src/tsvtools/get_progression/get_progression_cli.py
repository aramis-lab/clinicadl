from typing import List

import click

from clinicadl.utils import cli_param


@click.command(name="get-progression", no_args_is_help=True)
@cli_param.argument.data_tsv
@click.option(
    "--time_horizon",
    help="Time horizon to analyse stability of the label in the case of a progressive disease.",
    default=36,
    type=int,
)
def cli(data_tsv, time_horizon):
    """Get the progression of Alzheimer's disease.

    DATA_TSV is the path to the tsv file with columns including ["participants_id", "session_id"]

    TIME_HORIZON is the time in months chosen to analyse the stability of the label (default is 36)

    Outputs are stored in DATA_TSV.
    """

    from .get_progression import get_progression

    get_progression(data_tsv, horizon_time=time_horizon)


if __name__ == "__main__":
    cli()
