import logging

from clinicadl.callbacks.implemented.logger import (
    ConsoleFormatter,
    StdLevelFilter,
    setup_logging,
)

# -----------------------
# StdLevelFilter tests
# -----------------------


def test_std_level_filter_stdout():
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    )
    filt = StdLevelFilter(err=False)
    assert filt.filter(record) is True

    filt_err = StdLevelFilter(err=True)
    assert filt_err.filter(record) is False


def test_std_level_filter_stderr():
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    )
    filt = StdLevelFilter(err=False)
    assert filt.filter(record) is False

    filt_err = StdLevelFilter(err=True)
    assert filt_err.filter(record) is True


# -----------------------
# ConsoleFormatter tests
# -----------------------


def test_console_formatter_info_and_warning():
    fmt = ConsoleFormatter()
    info_record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="info",
        args=(),
        exc_info=None,
    )
    warning_record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg="warn",
        args=(),
        exc_info=None,
    )
    error_record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="error",
        args=(),
        exc_info=None,
    )

    info_str = fmt.format(info_record)
    warning_str = fmt.format(warning_record)
    error_str = fmt.format(error_record)

    assert "info" in info_str
    assert "warn" in warning_str
    assert "ERROR" in error_str  # fallback format contains levelname


# -----------------------
# setup_logging tests
# -----------------------


def test_setup_logging_returns_logger(tmp_path):
    logger = setup_logging(verbose=True)
    assert isinstance(logger, logging.Logger)
    # Check at least 2 handlers (stdout + stderr + optional file)
    assert len(logger.handlers) >= 2
