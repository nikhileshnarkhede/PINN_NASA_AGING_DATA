"""
utils/logger.py
===============
Central logging configuration for the project.

One call to setup_logging() at program start (typically from main.py) gives
every module a consistent console format. Modules themselves never configure
logging -- they just do `log = logging.getLogger(__name__)` and log normally;
this keeps logging setup in exactly one place.
"""
import logging
import sys

# Format shared by every module: LEVEL | logger name | message.
_LOG_FORMAT = "%(levelname)-8s | %(name)-22s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: int | str = logging.INFO) -> None:
    """
    Configure root logging for the whole project.

    Installs a single stdout handler with the shared format. Safe to call more
    than once -- any existing handlers are cleared first, so a second call does
    not produce duplicated log lines.

    Args:
        level: Logging level, as a logging constant (e.g. logging.INFO) or its
               name ("INFO", "DEBUG", ...).
    """
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers so repeated calls don't duplicate output.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)

    # Quieten noisy third-party loggers; the project's own logs stay at `level`.
    for noisy in ("matplotlib", "PIL", "git", "mlflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    A thin convenience wrapper over logging.getLogger so modules can import a
    single helper instead of the logging module directly. Either style works.

    Args:
        name: Logger name, conventionally the module's __name__.

    Returns:
        The logging.Logger for that name.
    """
    return logging.getLogger(name)
