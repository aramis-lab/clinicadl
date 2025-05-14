import logging
import sys
from pathlib import Path

from .base import Callback


class StdLevelFilter(logging.Filter):
    """TO COMPLETE"""

    def __init__(self, err=False):
        super().__init__()
        self.err = err

    def filter(self, record):
        if record.levelno <= logging.WARNING:
            return not self.err
        return self.err


# Create formatter for console
class ConsoleFormatter(logging.Formatter):
    """TO COMPLETE"""

    FORMATS = {
        logging.INFO: "%(asctime)s - %(message)s",
        logging.WARNING: "%(asctime)s - %(levelname)s: %(message)s",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, "%H:%M:%S")
        return formatter.format(record)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Setup ClinicaDL's logging facilities.
    Parameters
    ----------
    verbose: bool
        The desired level of verbosity for logging.
        (False (default): INFO, True: DEBUG)
    """
    logging_level = "DEBUG" if verbose else "INFO"

    # Define the module level logger.
    logger = logging.getLogger("clinicadl")
    logger.setLevel(logging_level)

    console_formatter = ConsoleFormatter()
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # est ce que les erreurs s'affichent sur la console
    err_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    err_handler = logging.StreamHandler(stream=sys.stderr)
    err_handler.addFilter(StdLevelFilter(err=True))
    err_handler.setFormatter(err_formatter)

    logger.handlers = []
    logger.addHandler(console_handler)
    logger.addHandler(err_handler)

    # Create file handler for debug mode with its own formatter and add it to the logger
    if verbose:
        debug_file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        debug_file_name = "clinicadl_debug.log"
        file_handler = logging.FileHandler(debug_file_name)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(debug_file_formatter)
        logger.addHandler(file_handler)
        logger.warning("Debug log will be saved at %s", (Path.cwd() / debug_file_name))

    return logger


class LoggerCallback(Callback):
    """TO COMPLETE"""

    def __init__(self, verbose: bool = False):
        self.logger = setup_logging(verbose=verbose)

    def on_train_begin(self, **kwargs):
        self.logger.info("Beginning of the training")
        self.logger.info("Training on %s", kwargs["device"])

    def on_train_end(self, **kwargs):
        self.logger.info("End of the training")

    def on_epoch_begin(self, epoch: int):
        self.logger.info("Beginning of epoch %s", epoch)

    def on_epoch_end(self, epoch: int):
        self.logger.info("Epoch %s completed", epoch)

    def on_batch_begin(self, batch: int):
        self.logger.info("Beginning of batch %s", batch)

    def on_batch_end(self, batch: int, **kwargs):
        self.logger.info("Batch %s completed", batch)
