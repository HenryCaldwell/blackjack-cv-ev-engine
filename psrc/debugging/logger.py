import logging
from logging import Logger


def setup_logger(name: str = "rain_vision", level: int = logging.DEBUG) -> Logger:
    """
    This method retrieves or creates a logger with the given name, adds a StreamHandler with a timestamped,
    level-prefixed formatter if the logger has no handlers yet, sets its logging level, and returns it.

    Parameters:
        name (str): The name of the logger.
        level (int): The logging level threshold.

    Returns:
        Logger: A configured Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger
