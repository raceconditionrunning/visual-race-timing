import logging
import sys


def is_debugger_attached():
    if 'pydevd' in sys.modules:
        # PyCharm debugger is attached
        return True
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


def get_logger(name=None, force_level=None):
    """
    Get a logger with automatic debug level detection.

    Args:
        name: Logger name (defaults to caller's __name__)
        force_level: Override automatic level detection
    """
    if force_level:
        log_level = force_level
    elif is_debugger_attached():
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Only configure basic config once
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    return logger


def configure_root_logging():
    """Configure root logger based on debugger state."""
    log_level = logging.DEBUG if is_debugger_attached() else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # Reconfigure if already configured
    )
    return log_level
