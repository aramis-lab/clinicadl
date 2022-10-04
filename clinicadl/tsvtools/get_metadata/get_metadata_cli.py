from typing import List

import click

from clinicadl.utils import cli_param


@click.command(name="get-metadata", no_args_is_help=True)
@cli_param.option.variables_of_interest
def cli(output_tsv, variables_of_interest):

    from .get_metadata import get_metadata

    if len(variables_of_interest) == 0:
        variables_of_interest = None

    results_directory = Path(output_tsv).parents[0]
    formatted_data_tsv = results_directory / "labels.tsv"

    metadata_df = merge_tsv_reader(formatted_data_tsv)
    output_df = merge_tsv_reader(output_tsv)

    result_df = get_metadata(metadata_df, output_df, variables_of_interest)

    result_df.to_csv(output_tsv, sep="\t")


if __name__ == "__main__":
    cli()
