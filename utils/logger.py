"""Shared logging configuration for AI Texture to PBR.

Usage:
    from ..utils.logger import get_logger
    log = get_logger(__name__)
    log.debug("detailed info")
    log.info("progress update")
    log.warning("non-fatal issue")
    log.error("something went wrong")

Debug output is controlled by the add-on preference `debug_mode`.
When disabled, only WARNING and above are printed.
"""

import logging
import sys

_loggers: dict[str, logging.Logger] = {}
_initialized: bool = False


def get_logger(name: str = "blender_ai") -> logging.Logger:
    """Return a logger for the given module name, creating it on first call.

    Log level defaults to WARNING; call `set_debug_mode(True)` to enable
    DEBUG output (typically from the add-on preferences handler).
    """
    global _initialized
    if not _initialized:
        _init_root_logger()
        _initialized = True

    if name in _loggers:
        return _loggers[name]

    logger_name = name if name.startswith("blender_ai") else f"blender_ai.{name}"
    logger = logging.getLogger(logger_name)
    _loggers[name] = logger
    return logger


def set_debug_mode(enabled: bool):
    """Toggle debug logging across all logger instances."""
    level = logging.DEBUG if enabled else logging.WARNING
    for logger in _loggers.values():
        logger.setLevel(level)


def _init_root_logger():
    """Configure a lightweight stderr handler so output appears in Blender's console."""
    root = logging.getLogger("blender_ai")
    root.setLevel(logging.WARNING)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "[%(name)s] %(levelname)s: %(message)s"
    ))
    root.addHandler(handler)
    root.propagate = False
