import click

from clinicadl.utils import cli_param

cmd_name = "analysis"


@click.command(name=cmd_name)
@cli_param.argument.merged_tsv
@cli_param.argument.formatted_data_directory
@cli_param.argument.results_directory
@cli_param.option.diagnoses
def cli(merged_tsv, formatted_data_directory, results_directory, diagnoses):
    """
    Produces a demographic analysis of the extracted labels.
    MERGED_TSV is the output of `clinica iotools merge-tsv`.
    MISSING_MODS_DIRECTORY is  the outputs of `clinicadl tsvtool getlabels`.
    Outputs are stored in RESULTS_TSV.
    """
    from .analysis import demographics_analysis

    demographics_analysis(
        merged_tsv, formatted_data_directory, results_directory, diagnoses
    )


if __name__ == "__main__":
    cli()