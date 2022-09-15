# coding: utf8

import logging
import sys

import click

from clinicadl.extract.extract_cli import cli as extract_cli
from clinicadl.generate.generate_cli import cli as generate_cli
from clinicadl.interpret.interpret_cli import cli as interpret_cli
from clinicadl.predict.predict_cli import cli as predict_cli
from clinicadl.quality_check.qc_cli import cli as qc_cli
from clinicadl.random_search.random_search_cli import cli as random_search_cli
from clinicadl.train.train_cli import cli as train_cli
from clinicadl.tsvtools.cli import cli as tsvtools_cli


class StdLevelFilter(logging.Filter):
    def __init__(self, err=False):
        super().__init__()
        self.err = err

    def filter(self, record):
        if record.levelno <= logging.INFO:
            return not self.err
        return self.err


def setup_logging(verbose: bool = False) -> None:
    """
    Setup ClinicaDL's logging facilities.
    Args:
        verbose: The desired level of verbosity for logging.
            (False (default): INFO, True: DEBUG)
    """
    logging_level = "DEBUG" if verbose else "INFO"

    # Define the module level logger.
    logger = logging.getLogger("clinicadl")
    logger.setLevel(logging_level)

    # Create formatter for console
    class ConsoleFormatter(logging.Formatter):

        FORMATS = {
            logging.INFO: "%(asctime)s - %(message)s",
            logging.WARNING: "%(asctime)s - %(name)s - %(levelname)s: %(message)s",
        }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt, "%H:%M:%S")
            return formatter.format(record)

    console_formatter = ConsoleFormatter()
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # est ce que les erreurs s'affichent sur la console
    err_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    err_handler = logging.StreamHandler(stream=sys.stderr)
    err_handler.addFilter(logging.StdLevelFilter(err=True))
    err_handler.setFormatter(err_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(err_handler)

    # Create file handler for debug mode with its own formatter and add it to the logger
    if verbose:
        debug_file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        file_handler = logging.FileHandler("clinicadl_debug.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(debug_file_formatter)
        logger.addHandler(file_handler)


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
cli.add_command(extract_cli)
cli.add_command(predict_cli)
cli.add_command(interpret_cli)
cli.add_command(qc_cli)
cli.add_command(random_search_cli)

if __name__ == "__main__":
    cli()
