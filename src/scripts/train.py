import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
# import json
# import numpy as np
from pathlib import Path
from app.logging_config import configure_logging
from scripts.agent import build_trainer_from_config, load_config
# from app.agent_utils import load_agent

parser = argparse.ArgumentParser(description='Train Agent')
parser.add_argument(
    '--config',
    type=str,
    required=True,
    help='Path to the agent configuration file')

# parser.add_argument(
#     '--log_level',
#     type=str,
#     required=False,
#     default='INFO',
#     help='Logging level')
args = parser.parse_args()
config_file = Path(args.config)
config = load_config(config_file)
log_level = config.get('log_level', 'INFO')

# Create logger
logger = configure_logging(log_level, log_dir=config.get('save_dir'))

def main(config):
    try:
        trainer = build_trainer_from_config(config)
        trainer.train()
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {str(e)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    main(config)
