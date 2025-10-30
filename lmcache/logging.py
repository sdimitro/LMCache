# SPDX-License-Identifier: Apache-2.0
# Standard
from datetime import datetime
from logging import Logger
import logging
import os
import sys


def build_format(color):
    reset = "\x1b[0m"
    underline = "\x1b[3m"
    return (
        f"{color}[%(asctime)s] LMCache %(levelname)s:{reset} %(message)s "
        f"{underline}(%(filename)s:%(lineno)d:%(name)s){reset}"
    )


class CustomFormatter(logging.Formatter):
    grey = "\x1b[1m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: build_format(grey),
        logging.INFO: build_format(green),
        logging.WARNING: build_format(yellow),
        logging.ERROR: build_format(red),
        logging.CRITICAL: build_format(bold_red),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_log_level() -> int:
    """
    Try to read LMCACHE_LOG_LEVEL from environment variables.
    Could be:
    - DEBUG
    - INFO
    - WARNING
    - ERROR
    - CRITICAL

    If not found, defaults to INFO.
    """
    log_level = os.getenv("LMCACHE_LOG_LEVEL", "INFO").upper()
    return getattr(logging, log_level, logging.INFO)


def init_logger(name: str) -> Logger:
    # Get the logger
    logger = logging.getLogger(name)

    # Clear any existing handlers
    logger.handlers.clear()

    # Prevent propagation to parent loggers
    logger.propagate = False

    # Add our custom handler
    ch = logging.StreamHandler()
    ch.setLevel(get_log_level())
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    logger.setLevel(get_log_level())
    return logger


# Global variable to store the loguru instance (singleton pattern)
_loguru_instance = None
_loguru_configured = False


def get_loguru():
    """
    Get or initialize a loguru logger instance.

    This function returns a singleton loguru logger that is configured
    on first access. The logger respects the LMCACHE_LOG_LEVEL environment
    variable and provides colorized output similar to the standard logger.

    Returns:
        loguru.Logger: The configured loguru logger instance
    """
    global _loguru_instance, _loguru_configured

    if _loguru_instance is None:
        # Third Party
        from loguru import logger

        _loguru_instance = logger

    if not _loguru_configured:
        # Remove default handler
        _loguru_instance.remove()
        logger.level("INFO", color="<green>")

        log_level = os.getenv("LMCACHE_NULOG_LEVEL", "INFO").upper()
        log_format = (
            "<level>[{time:YYYY-MM-DD HH:mm:ss,SSS}] LMCache {level}:</level> "
            "{message} <dim>({file}:{line}:{name})</dim>"
        )

        _loguru_instance.add(
            sys.stderr,
            format=log_format,
            level=log_level,
            colorize=True,
        )

        pid = os.getpid()
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        default_log_dir = os.path.expanduser(f"~/.local/state/lmcache/logs/{timestamp}")
        log_dir = os.getenv("LMCACHE_NULOG_DIR", default_log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, mode=0o775, exist_ok=True)
        _loguru_instance.add(
            os.path.join(log_dir, f"lmcache-{pid}.log"),
            format=log_format,
            level=log_level,
            rotation="100 MB",
            retention="30 days",
            compression="gz",
        )

        _loguru_configured = True

    return _loguru_instance


if __name__ == "__main__":
    # Test standard logger
    logger = init_logger(__name__)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    # Test loguru logger
    print("\n--- Testing Loguru Logger ---")
    loguru = get_loguru()
    loguru.debug("Loguru debug message")
    loguru.info("Loguru info message")
    loguru.warning("Loguru warning message")
    loguru.error("Loguru error message")
    loguru.critical("Loguru critical message")

# import logging
# from logging import Logger
#
# logging.basicConfig(
#    format="\033[33m%(levelname)s LMCache: \033[0m%(message)s "
#    "[%(asctime)s] -- %(pathname)s:%(lineno)d",
#    level=logging.INFO,
# )
#
#
# def init_logger(name: str) -> Logger:
#    logger = logging.getLogger(name)
#    logger.setLevel(logging.DEBUG)
#    return logger
