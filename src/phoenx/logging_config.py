import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGING_CONFIGURED = False
_LOGGER_NAME = "phoenx"


def _set_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def configure_logging(level: str = "INFO", log_dir: str | Path | None = None) -> logging.Logger:
    global _LOGGING_CONFIGURED

    phoenx_logger = logging.getLogger(_LOGGER_NAME)
    phoenx_logger.setLevel(_set_level(level))
    phoenx_logger.propagate = False

    if _LOGGING_CONFIGURED:
        return phoenx_logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Configure logging path
    if log_dir is not None:
        log_path = Path(log_dir) / "phoenx.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_path = Path(__file__).resolve().parents[1] / "phoenx.log"

    # Configure handler
    file_handler = RotatingFileHandler(
        log_path,
        mode="w",
        # maxBytes=1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.NOTSET)
    file_handler.setFormatter(formatter)

    phoenx_logger.handlers.clear()
    
    phoenx_logger.addHandler(file_handler)

    _LOGGING_CONFIGURED = True
    return phoenx_logger


def get_logger(name: str | None = None, level: str | int | None = None) -> logging.Logger:
    phoenx_logger = logging.getLogger(_LOGGER_NAME)
    logger = phoenx_logger if not name else phoenx_logger.getChild(name)

    if level is None:
        logger.setLevel(logging.NOTSET)
    else:
        logger.setLevel(_set_level(level))

    return logger