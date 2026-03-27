import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGING_CONFIGURED = False
_APP_LOGGER_NAME = "app"


def _set_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def configure_logging(level: str = "INFO") -> logging.Logger:
    global _LOGGING_CONFIGURED

    app_logger = logging.getLogger(_APP_LOGGER_NAME)
    app_logger.setLevel(_set_level(level))
    app_logger.propagate = False

    if _LOGGING_CONFIGURED:
        return app_logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.NOTSET)
    console_handler.setFormatter(formatter)

    log_path = Path(__file__).resolve().parents[1] / "app.log"
    file_handler = RotatingFileHandler(
        log_path,
        mode="w",
        maxBytes=1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.NOTSET)
    file_handler.setFormatter(formatter)

    app_logger.handlers.clear()
    app_logger.addHandler(console_handler)
    app_logger.addHandler(file_handler)

    _LOGGING_CONFIGURED = True
    return app_logger


def get_logger(name: str | None = None, level: str | int | None = None) -> logging.Logger:
    app_logger = logging.getLogger(_APP_LOGGER_NAME)
    logger = app_logger if not name else app_logger.getChild(name)

    if level is None:
        logger.setLevel(logging.NOTSET)
    else:
        logger.setLevel(_set_level(level))

    return logger