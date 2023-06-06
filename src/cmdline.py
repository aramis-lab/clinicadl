# coding: utf8

import click

from src.pipelines.generate.generate_cli import cli as generate_cli
from src.pipelines.interpret.interpret_cli import cli as interpret_cli
from src.pipelines.predict.predict_cli import cli as predict_cli
from src.pipelines.prepare_data.prepare_data_cli import cli as prepare_data_cli
from src.pipelines.quality_check.qc_cli import cli as qc_cli
from src.pipelines.random_search.random_search_cli import cli as random_search_cli
from src.pipelines.train.train_cli import cli as train_cli
from src.pipelines.tsvtools.cli import cli as tsvtools_cli
from src.utils.logger import setup_logging

CONTEXT_SETTINGS = dict(
    # Extend content width to avoid shortening of pipeline help.
    max_content_width=160,
    # Display help string with -h, in addition to --help.
    help_option_names=["-h", "--help"],
)


@click.group(context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
@click.version_option()
@click.option("-v", "--verbose", is_flag=True, help="Verbosity mode.")
def cli(verbose):
    """ClinicaDL command line.

    For more information please read the doc: https://clinicadl.readthedocs.io/en/latest/ .
    Source code is available on GitHub: https://github.com/aramis-lab/clinicaDL .

    Do not hesitate to create an issue to report a bug or suggest an improvement.
    """
    setup_logging(verbose=verbose)


cli.add_command(tsvtools_cli)
cli.add_command(train_cli)
cli.add_command(generate_cli)
cli.add_command(prepare_data_cli)
cli.add_command(predict_cli)
cli.add_command(interpret_cli)
cli.add_command(qc_cli)
cli.add_command(random_search_cli)

if __name__ == "__main__":
    cli()
