import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import json
from pathlib import Path
from app.logging_config import configure_logging
from scripts.agent import build_trainer_from_config, load_config
from app.trainer import Trainer


parser = argparse.ArgumentParser(description='Train Agent')
parser.add_argument(
    '--config',
    type=str,
    required=False,
    help='Path to the agent configuration file')

parser.add_argument(
    '--agent_dir',
    type=str,
    required=False,
    help='Path to the agent directory. Can be used to resume training'
)

parser.add_argument(
    '--log_level',
    type=str,
    required=False,
    help='Logging level'
)

args = parser.parse_args()
if args.config is not None:
    agent_dir = None
    config = load_config(Path(args.config))
    log_level = args.log_level if args.log_level is not None else config.get('log_level', 'INFO')

elif args.agent_dir is not None:
    agent_dir = Path(args.agent_dir)
    log_level = args.log_level
    if log_level is None:
        with open(agent_dir / "config.json", encoding="utf-8") as f:
            config = json.load(f)
        log_level = config.get('log_level', 'INFO')
else:
    raise ValueError('Either --config or --agent_dir must be provided.')



# Create logger
logger = configure_logging(log_level, log_dir=config.get('save_dir'))

def main(agent_dir: Path | None = None, config: dict | None = None, log_level: str | None = None):
    try:
        if agent_dir is not None:
            trainer = Trainer.load(agent_dir, load_buffer=True, log_level=log_level)
        elif config is not None:
            trainer = build_trainer_from_config(config, log_level)
        trainer.train()
    except FileNotFoundError as e:
        logger.error(f"Configuration file or agent directory not found: {str(e)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    main(agent_dir=agent_dir, config=config, log_level=log_level)
