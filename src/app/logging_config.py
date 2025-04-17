import logging
import os

# Clear the debug.log file if it exists
log_file = 'debug.log'
if os.path.exists(log_file):
    with open(log_file, 'w'):
        pass

# Create a custom logger for our application
logger = logging.getLogger('rl_agents')
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels for our application

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(log_file)

# Set levels for handlers
console_handler.setLevel(logging.INFO)  # Show INFO and above in console
file_handler.setLevel(logging.DEBUG)  # Log everything to file

# Create formatters and add them to the handlers
console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(console_format)
file_handler.setFormatter(file_format)

# Clear any existing handlers
if logger.hasHandlers():
    logger.handlers.clear()

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Set up root logger to WARNING level to suppress library debug messages
logging.getLogger().setLevel(logging.WARNING)

# Set specific loggers to WARNING level to suppress their debug messages
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('ray').setLevel(logging.INFO)  # Keep Ray at INFO level for important messages