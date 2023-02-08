import click

from clinicadl.utils import cli_param


@click.command(name="analysis", no_args_is_help=True)
@cli_param.argument.merged_tsv
@cli_param.argument.data_tsv
@cli_param.argument.results_tsv
@cli_param.option.diagnoses
def cli(merged_tsv, data_tsv, results_tsv, diagnoses):
    """Demographic analysis of the extracted labels.

    MERGED_TSV is the output of `clinica iotools merge-tsv`.

    DATA_TSV is the output of `clinicadl tsvtools get-labels`.

    Results are stored in RESULTS_TSV.
    """
    from .analysis import demographics_analysis

    demographics_analysis(merged_tsv, data_tsv, results_tsv, diagnoses)


if __name__ == "__main__":
    cli()
