# coding: utf8

import click

from clinicadl.train.train_cli import cli as train_cli
from clinicadl.generate.cli import cli as generate_cli
from clinicadl.extract.extract_cli import cli as extract_cli
from clinicadl.quality_check.cli import cli as qc_cli
from clinicadl.predict.predict_cli import cli as predict_cli
from clinicadl.interpret.interpret_cli import cli as interpret_cli
from clinicadl.tsvtools.cli import cli as tsvtools_cli
from clinicadl.random_search.random_search_cli import cli as random_search_cli

CONTEXT_SETTINGS = dict(
    # Extend content width to avoid shortening of pipeline help.
    max_content_width=160,
    # Display help string with -h, in addition to --help.
    help_option_names=["-h", "--help"],
)


def setup_logging(verbosity: int = 0) -> None:
    """
    Setup Clinicadl's logging facilities.
    Args:
        verbosity (int): The desired level of verbosity for logging.
            (0 (default): WARNING, 1: INFO, 2: DEBUG)
    """
    from logging import DEBUG, INFO, WARNING, getLogger, StreamHandler, Formatter
    from sys import stdout


    # Cap max verbosity level to 2.
    verbosity = min(verbosity, 2)

    # Define the module level logger.
    logger = getLogger("clinicadl")
    logger.setLevel([WARNING, INFO, DEBUG][verbosity])

    # Add console handler
    console_handler = StreamHandler(stdout)
    # create formatter
    formatter = Formatter("%(asctime)s - %(levelname)s: %(message)s", "%H:%M:%S")
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option()
@click.option(
    "-v", "--verbose", "verbosity", count=True, help="Increase logging verbosity."
)
def cli(verbosity):
    setup_logging(verbosity=verbosity)


cli.add_command(tsvtools_cli)
cli.add_command(train_cli)
cli.add_command(generate_cli)
cli.add_command(extract_cli)
cli.add_command(predict_cli)
cli.add_command(interpret_cli)
cli.add_command(qc_cli)
cli.add_command(random_search_cli)

if __name__ == "__main__":
    cli()
