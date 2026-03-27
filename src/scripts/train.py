import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
# import json
# import numpy as np
from pathlib import Path
from app.logging_config import configure_logging
from scripts.agent import build_trainer_from_config_path
# from app.agent_utils import load_agent

parser = argparse.ArgumentParser(description='Train Agent')
parser.add_argument(
    '--config',
    type=str,
    required=True,
    help='Path to the agent configuration directory')

parser.add_argument(
    '--log_level',
    type=str,
    required=False,
    default='INFO',
    help='Logging level')
args = parser.parse_args()
config_dir = Path(args.config)
log_level = args.log_level.upper()

# Create logger
logger = configure_logging(log_level)

def main(config_dir):
    try:
        trainer = build_trainer_from_config_path(config_dir)
        trainer.train()
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {str(e)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    main(config_dir)
