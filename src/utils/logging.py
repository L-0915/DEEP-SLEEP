"""Logging setup utilities for the DeepSleep project.

Provides a factory function for configuring loggers with consistent formatting,
including both console (stdout) and file handlers.  All training scripts and
utility modules should obtain their loggers through ``setup_logger`` so that
output is uniform and configurable from a single location.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track already-configured names to avoid adding duplicate handlers.
_configured_loggers: set[str] = set()


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    force: bool = False,
) -> logging.Logger:
    """Configure and return a logger with console and optional file handlers.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
        level: Logging level (default ``logging.INFO``).
        log_file: Optional path to a log file.  If provided a
            ``RotatingFileHandler`` is attached.  When *None*, only a
            ``StreamHandler`` writing to *stdout* is used.
        force: If ``True``, reconfigure the logger even if it was already set
            up (useful for tests or interactive sessions).

    Returns:
        A fully configured ``logging.Logger`` instance.
    """
    if not force and name in _configured_loggers:
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove any pre-existing handlers (relevant when *force* is True).
    logger.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # -- Console handler (stdout) --
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # -- Optional file handler --
    if log_file is not None:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to the root logger to avoid duplicate messages.
    logger.propagate = False

    _configured_loggers.add(name)
    return logger
