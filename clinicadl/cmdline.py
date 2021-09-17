# coding: utf8

import click

from clinicadl.extract.extract_cli import cli as extract_cli
from clinicadl.generate.generate_cli import cli as generate_cli
from clinicadl.interpret.interpret_cli import cli as interpret_cli
from clinicadl.predict.predict_cli import cli as predict_cli
from clinicadl.quality_check.qc_cli import cli as qc_cli
from clinicadl.random_search.random_search_cli import cli as random_search_cli
from clinicadl.save_tensor.save_tensor_cli import cli as save_tensor_cli
from clinicadl.train.resume_cli import cli as resume_cli
from clinicadl.train.train_cli import cli as train_cli
from clinicadl.tsvtools.cli import cli as tsvtools_cli
from clinicadl.utils.maps_manager.logwriter import setup_logging

CONTEXT_SETTINGS = dict(
    # Extend content width to avoid shortening of pipeline help.
    max_content_width=160,
    # Display help string with -h, in addition to --help.
    help_option_names=["-h", "--help"],
)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option()
@click.option(
    "-v", "--verbose", "verbosity", count=True, help="Increase logging verbosity."
)
def cli(verbosity):
    """ClinicaDL command line.

    For more information please read the doc: https://clinicadl.readthedocs.io/en/latest/ .
    Source code is available on GitHub: https://github.com/aramis-lab/clinicaDL .

    Do not hesitate to create an issue to report a bug or suggest an improvement.
    """
    setup_logging(verbosity=verbosity)


cli.add_command(tsvtools_cli)
cli.add_command(train_cli)
cli.add_command(resume_cli)
cli.add_command(generate_cli)
cli.add_command(extract_cli)
cli.add_command(predict_cli)
cli.add_command(interpret_cli)
cli.add_command(qc_cli)
cli.add_command(random_search_cli)
cli.add_command(save_tensor_cli)

if __name__ == "__main__":
    cli()
