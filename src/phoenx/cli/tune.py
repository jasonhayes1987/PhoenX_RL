"""Command-line entry point for ``phoenx-tune``, the Ray Tune sweep driver.

Mirrors ``phoenx.cli.train``'s ``argparse`` + ``configure_logging`` shape.
Loading Ray only happens once a sweep actually runs (or is validated);
``phoenx.ray_tune`` itself imports Ray at module scope, so it is never
imported from ``phoenx/__init__.py`` or any other always-loaded module —
only from here.
"""

import argparse

from phoenx.logging_config import configure_logging
from phoenx.ray_tune import load_sweep_config, run_sweep, validate_only


def main() -> None:
    """Parse CLI flags, load a sweep config, and run (or validate) it.

    Flags:
        ``--config``: Path to the sweep configuration YAML, or a bundled
            example name under ``phoenx/examples/sweeps/`` (required).
        ``--base-config``: Overrides ``sweep['base_config']``.
        ``--num-samples``: Overrides every phase's ``tune.num_samples``.
        ``--max-concurrent``: Overrides every phase's
            ``max_concurrent_trials``.
        ``--name``: Sweep identifier; artifacts land under
            ``<storage-path>/<name>/<phase>/``.
        ``--storage-path``: Root directory for Ray Tune experiment storage
            (default ``"ray_results"`` under the current working directory).
        ``--from-phase``: Resume a multi-phase sweep starting at this phase
            name, loading the preceding phase's promoted ``best_config.yml``.
        ``--resume``: Resume the starting phase's own interrupted ``Tuner``
            run if a matching experiment already exists on disk.
        ``--validate-only``: Sample and resolve trial configs per phase and
            print the diff, without touching Ray's cluster, then exit.
        ``--log-level``: Logging level (default ``"INFO"``).

    Raises:
        FileNotFoundError: If ``--config`` (or an overriding ``--base-config``)
            names neither an on-disk path nor a bundled example.
        ValueError: Propagated from sweep validation, phase resolution, or
            an unknown ``--from-phase`` name.
    """
    parser = argparse.ArgumentParser(description="Run a PhoenX Ray Tune hyperparameter sweep")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the sweep configuration YAML, or a bundled example name",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        required=False,
        help="Override the sweep's base_config path or bundled example name",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=False,
        help="Override every phase's tune.num_samples",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        required=False,
        help="Override every phase's max_concurrent_trials",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        help="Sweep name; artifacts land under <storage-path>/<name>/<phase>/",
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        required=False,
        help="Root directory for Ray Tune experiment storage (default: ./ray_results)",
    )
    parser.add_argument(
        "--from-phase",
        type=str,
        required=False,
        help="Resume a multi-phase sweep starting at this phase name",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the starting phase's own interrupted Tuner run if one exists on disk",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Sample and resolve trial configs per phase and print the diff; never touches Ray",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        required=False,
        help="Logging level",
    )

    args = parser.parse_args()

    log_level = args.log_level if args.log_level is not None else "INFO"
    logger = configure_logging(log_level, log_dir=args.storage_path)

    sweep = load_sweep_config(args.config)
    if args.base_config is not None:
        sweep["base_config"] = args.base_config

    if args.validate_only:
        validate_only(sweep, num_samples=args.num_samples if args.num_samples is not None else 3)
        return

    overrides = {}
    if args.num_samples is not None:
        overrides["num_samples"] = args.num_samples
    if args.max_concurrent is not None:
        overrides["max_concurrent_trials"] = args.max_concurrent

    result = run_sweep(
        sweep,
        storage_path=args.storage_path,
        sweep_name=args.name,
        from_phase=args.from_phase,
        resume=args.resume,
        **overrides,
    )
    logger.info("Sweep complete. Final config written to %s", result["final_config_path"])


if __name__ == "__main__":
    main()
