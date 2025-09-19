# Note: I don't know if the logger need to be in a callback ? WDYT ?

import logging
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from clinicadl.callbacks.training_state import _TrainingState

from .base import Callback


class StdLevelFilter(logging.Filter):
    """
    Logging filter to route messages based on severity level.

    This filter allows messages up to WARNING to go to stdout (if `err=False`),
    and ERROR and above to go to stderr (if `err=True`).

    Parameters
    ----------
    err : bool
        If True, allows only ERROR and above.
        If False, allows only messages of level WARNING and below.
    """

    def __init__(self, err: bool = False):
        super().__init__()
        self.err = err

    def filter(self, record: logging.LogRecord) -> bool:
        return not self.err if record.levelno <= logging.WARNING else self.err


class ConsoleFormatter(logging.Formatter):
    """
    Custom formatter for console output.

    Provides simplified format for INFO and WARNING messages,
    and falls back to the default format for others.
    """

    FORMATS = {
        logging.INFO: "%(asctime)s - %(message)s",
        logging.WARNING: "%(asctime)s - %(levelname)s: %(message)s",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(
            record.levelno, "%(asctime)s - %(levelname)s: %(message)s"
        )
        formatter = logging.Formatter(log_fmt, "%H:%M:%S")
        return formatter.format(record)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Setup ClinicaDL's logging facilities.

    Configures the logger for the 'clinicadl' namespace to print messages to
    stdout/stderr and optionally to a debug file.

    Parameters
    ----------
    verbose : bool, default=False
        If True, enable DEBUG logging and write logs to a file.

    Returns
    -------
    logging._Logger
        Configured logger instance.
    """
    logging_level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger("clinicadl")
    logger.setLevel(logging_level)
    logger.handlers = []  # Clear existing handlers

    # Standard output handler (INFO, WARNING)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ConsoleFormatter())
    logger.addHandler(console_handler)

    # Standard error handler (ERROR and above)
    err_handler = logging.StreamHandler(stream=sys.stderr)
    err_handler.addFilter(StdLevelFilter(err=True))
    err_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    err_handler.setFormatter(err_formatter)
    logger.addHandler(err_handler)

    # Optional file handler (DEBUG and above)
    if verbose:
        debug_log_path = Path("clinicadl_debug.log")
        file_handler = logging.FileHandler(debug_log_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.warning("Debug log will be saved at %s", debug_log_path.resolve())

    return logger


class _Logger(Callback):
    """
    Callback that logs major training events to console and/or file.

    Parameters
    ----------
    verbose : bool, default=False
        If True, enables detailed DEBUG-level logging.
    """

    def __init__(self, verbose: bool = True):
        self.logger = setup_logging(verbose=verbose)
        self.train_progress_bar = None

    def on_train_begin(self, config: _TrainingState, **kwargs) -> None:
        self.logger.info("Beginning of the training for split %s", config.split.index)
        self.logger.info("Training on %s", config.comp.device)

    def on_train_end(self, config: _TrainingState, **kwargs) -> None:
        self.logger.info("End of the training")

    def on_epoch_begin(self, config: _TrainingState, **kwargs) -> None:
        train_loader = config.split.train_loader
        rank = kwargs.pop("rank", -1)
        if train_loader is not None and (rank == 0 or rank == -1):
            now = datetime.now().strftime("%H:%M:%S")
            self.train_progress_bar = tqdm(
                total=len(train_loader),
                unit="batch",
                desc=f"{now} - Training of epoch {config.epoch}/{config.optim.epochs - 1}",
            )

    def on_epoch_end(self, config: _TrainingState, **kwargs) -> None:
        if self.train_progress_bar is not None:
            self.train_progress_bar.close()

    def on_batch_begin(self, config: _TrainingState, **kwargs) -> None:
        self.logger.debug("Beginning of batch %d", config.batch)

    def on_batch_end(self, config: _TrainingState, **kwargs) -> None:
        self.logger.debug("Batch %d completed", config.batch)

        if self.train_progress_bar is not None:
            self.train_progress_bar.update(1)
