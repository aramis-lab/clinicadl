from typing import List

import click

from clinicadl.utils import cli_param


@click.command(name="get-metadata", no_args_is_help=True)
@cli_param.argument.formatted_data_tsv
@cli_param.argument.output_tsv
@cli_param.option.variables_of_interest
def cli(formatted_data_tsv, output_tsv, variables_of_interest):

    from .get_metadata import get_metadata

    if len(variables_of_interest) == 0:
        variables_of_interest = None

    get_metadata(formatted_data_tsv, output_tsv, variables_of_interest)


if __name__ == "__main__":
    cli()
