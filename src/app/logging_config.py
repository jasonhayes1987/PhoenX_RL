import logging
from logging.handlers import RotatingFileHandler
import os

# Clear the debug.log file if it exists
# log_file = 'debug.log'
# if os.path.exists(log_file):
#     with open(log_file, 'w'):
#         pass


def get_logger(name, level='debug'):
    logger = logging.getLogger(name)

    if not logger.handlers:
        if level == 'debug':
            logger.setLevel(logging.DEBUG)
        elif level == 'info':
            logger.setLevel(logging.INFO)
        elif level == 'warning':
            logger.setLevel(logging.WARNING)
        elif level == 'error':
            logger.setLevel(logging.ERROR)

        # Create handlers
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        file_handler = RotatingFileHandler('app.log', maxBytes=1024*1024, backupCount=5)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        # Set levels for handlers
        # console_handler.setLevel(logging.INFO)  # Show INFO and above in console
        # file_handler.setLevel(logging.DEBUG)  # Log everything to file


        # Set up root logger to WARNING level to suppress library debug messages
        logging.getLogger().setLevel(logging.WARNING)

        # Set specific loggers to WARNING level to suppress their debug messages
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('numba').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('ray').setLevel(logging.INFO)  # Keep Ray at INFO level for important messages

    return logger
