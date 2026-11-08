import argparse
from pathlib import Path
import json
import logging

from phoenx.builder import build_trainer_from_config, load_config
from phoenx.logging_config import configure_logging
from phoenx.trainer import Trainer

def main() -> None:
    parser = argparse.ArgumentParser(description='Train a PhoenX Agent')
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
    
    if agent_dir is not None:
        trainer = Trainer.load(agent_dir, load_buffer=True, log_level=log_level)
    elif config is not None:
        trainer = build_trainer_from_config(config, log_level)
    else:
        logger.error("Configuration file or agent directory not found")
        raise FileNotFoundError("Configuration file or agent directory not found")

    trainer.train()

if __name__ == "__main__":
    main()
