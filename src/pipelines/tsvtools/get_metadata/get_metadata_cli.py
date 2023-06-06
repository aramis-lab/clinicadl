import click

from clinicadl.utils import cli_param


@click.command(name="get-metadata", no_args_is_help=True)
@cli_param.argument.data_tsv
@cli_param.argument.merged_tsv
@cli_param.option.variables_of_interest
def cli(data_tsv, merged_tsv, variables_of_interest):
    """Writes additional data in the tsv file.

    DATA_TSV is the path to the TSV file with colmuns including ["participants_id", "session_id"]

    MERGED_TSV is the path to the TSV file with all the data (output of clinica merge-tsv/ clinicadl get-labels)

    VARIABLES_OF_INTEREST is a list of variables (columns) that will be added to the tsv file

    Outputs are written in DATA_TSV.
    """

    from .get_metadata import get_metadata

    if len(variables_of_interest) == 0:
        variables_of_interest = None

    get_metadata(
        data_tsv=data_tsv,
        merged_tsv=merged_tsv,
        variables_of_interest=variables_of_interest,
    )


if __name__ == "__main__":
    cli()
