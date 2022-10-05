import click

from clinicadl.utils import cli_param


@click.command(name="analysis", no_args_is_help=True)
@cli_param.argument.merged_tsv
@cli_param.argument.formatted_data_tsv
@cli_param.argument.results_tsv
@cli_param.option.diagnoses
def cli(merged_tsv, formatted_data_tsv, results_tsv, diagnoses):
    """Demographic analysis of the extracted labels.

    MERGED_TSV is the output of `clinica iotools merge-tsv`.

    FORMATTED_DATA_TSV is the output of `clinicadl tsvtool getlabels`.

    Results are stored in RESULTS_DIRECTORY.
    """
    from .analysis import demographics_analysis

    demographics_analysis(merged_tsv, formatted_data_tsv, results_tsv, diagnoses)


if __name__ == "__main__":
    cli()
