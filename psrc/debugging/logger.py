"""
Module for setting up a custom logger.

This module defines a function, setup_logger, which configures a logger with a custom format and specified log
level. It ensures that multiple handlers are not added if the logger is already configured, providing a clean and
consistent logging setup for the application.
"""

import logging
from logging import Logger


def setup_logger(name: str = "rain_vision", level: int = logging.DEBUG) -> Logger:
    """
    Set up and return a logger with the specified name and logging level.

    Configures a logger to output log messages to the console using a specific format. It ensures that multiple
    handlers are not added to the same logger if it is already configured.

    Parameters:
      name (str): The name of the logger. Defaults to 'rain_vision'.
      level (int): The logging level to use (e.g., logging.DEBUG, logging.INFO). Defaults to logging.DEBUG.

    Returns:
      logger (logging.Logger): The configured logger instance.
    """
    logger = logging.getLogger(name)

    #  Prevent adding multiple handlers if logger is already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )  # Define the log message format including the timestamp, log level, logger name, and message
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(
        level
    )  # Set the logging level for the logger (e.g., DEBUG, INFO, WARNING, etc.)
    logger.propagate = False
    return logger
