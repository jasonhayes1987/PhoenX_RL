"""``phoenx-train``: train from a YAML config, or resume a saved run."""

import argparse
from pathlib import Path
import json

from phoenx.builder import build_trainer_from_config, load_config
from phoenx.logging_config import configure_logging
from phoenx.run_info import read_run_info
from phoenx.trainer import Trainer

def main() -> None:
    """Parse arguments, build or load the Trainer, and train."""
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
    parser.add_argument(
        '--stop_units',
        type=int,
        required=False,
        help="Override schedule.stop_units: the total training length, in the run's "
             "stop_unit (timesteps or episodes). With --agent_dir, use it to train "
             "further than the original run's limit."
    )
    args = parser.parse_args()

    if args.config is not None:
        agent_dir = None
        config = load_config(Path(args.config))
    elif args.agent_dir is not None:
        agent_dir = Path(args.agent_dir)
        with open(agent_dir / "config.json", encoding="utf-8") as f:
            config = json.load(f)
    else:
        raise ValueError('Either --config or --agent_dir must be provided.')
    log_level = args.log_level if args.log_level is not None else config.get('log_level', 'INFO')

    if agent_dir is not None and args.stop_units is not None:
        info = read_run_info(agent_dir)
        if info["progress"] is not None and args.stop_units <= info["progress"]:
            raise ValueError(
                f"--stop_units {args.stop_units} is not past the saved checkpoint, which is at "
                f"{info['stop_unit']} {info['progress']}; training would stop immediately."
            )

    # Create logger
    logger = configure_logging(log_level, log_dir=config.get('save_dir'))

    if agent_dir is not None:
        trainer = Trainer.load(agent_dir, load_buffer=True, log_level=log_level)
    else:
        trainer = build_trainer_from_config(config, log_level)

    if args.stop_units is not None:
        logger.info("stop_units overridden: %s -> %s",
                    trainer.schedule.stop_units, args.stop_units)
        trainer.schedule.stop_units = args.stop_units

    trainer.train()


if __name__ == "__main__":
    main()
