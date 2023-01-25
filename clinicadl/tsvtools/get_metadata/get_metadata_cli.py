from pathlib import Path
from typing import List

import click

from clinicadl.utils import cli_param
from clinicadl.utils.tsvtools_utils import merged_tsv_reader


@click.command(name="get-metadata", no_args_is_help=True)
@cli_param.argument.data_tsv
@cli_param.option.variables_of_interest
def cli(data_tsv, variables_of_interest):
    """Writes additional data in the tsv file.

    DATA_TSV is the path to the tsv file with colmuns including ["participants_id", "session_id"]

    VARIABLES_OF_INTEREST is a list of variables (columns) that will be added to the tsv file

    Outputs are written in DATA_TSV.
    """

    from .get_metadata import get_metadata

    if len(variables_of_interest) == 0:
        variables_of_interest = None

    results_directory = Path(data_tsv).parents[0]
    labels_tsv = results_directory / "labels.tsv"

    metadata_df = merged_tsv_reader(labels_tsv)
    output_df = merged_tsv_reader(data_tsv)

    result_df = get_metadata(metadata_df, output_df, variables_of_interest)

    result_df.to_csv(data_tsv, sep="\t")


if __name__ == "__main__":
    cli()
