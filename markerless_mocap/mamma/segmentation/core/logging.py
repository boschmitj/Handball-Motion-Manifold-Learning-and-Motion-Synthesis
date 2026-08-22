"""Centralized logging configuration using loguru.

INFO and below go to stdout (.out on cluster).
WARNING and above go to stderr (.err on cluster).
All levels are also saved to a log file when enable_file_logging() is called.

Usage::

    from core.logging import logger, enable_file_logging

    enable_file_logging("/path/to/output/run.log")
    logger.info("Processing camera IOI_09")
    logger.warning("Camera not found")
    logger.error("Segmentation failed")
"""
import sys
from loguru import logger

# Remove default handler (stderr for all levels)
logger.remove()

_CONSOLE_FORMAT = "<green>{time:HH:mm:ss}</green> | <cyan>[ma_masks]</cyan> <level>{level:<7}</level> | {message}"
_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | [ma_masks] {level:<7} | {message}"

# INFO and below -> stdout (visible in .out on cluster)
logger.add(
    sys.stdout,
    level="DEBUG",
    filter=lambda record: record["level"].no < 30,  # 30 = WARNING
    format=_CONSOLE_FORMAT,
)

# WARNING and above -> stderr (visible in .err on cluster)
logger.add(
    sys.stderr,
    level="WARNING",
    format=_CONSOLE_FORMAT,
)

_file_handler_id = None


def enable_file_logging(log_path):
    """Add a file handler that saves all log levels to a text file.

    Call once at the start of a run with the output directory path.
    Safe to call multiple times — replaces the previous file handler.
    """
    global _file_handler_id
    import os
    os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)

    # Remove previous file handler if any
    if _file_handler_id is not None:
        try:
            logger.remove(_file_handler_id)
        except ValueError:
            pass

    _file_handler_id = logger.add(
        log_path,
        level="DEBUG",
        format=_FILE_FORMAT,
        mode="w",  # overwrite per run
    )
    logger.info(f"Log file: {log_path}")
