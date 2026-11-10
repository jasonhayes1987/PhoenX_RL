"""Summarize a saved training run directory.

A run directory is what ``Trainer.save`` writes (the run's ``save_dir``)::

    <run_dir>/config.json        # the whole run's config
    <run_dir>/agent/             # weights, optimizers, normalizers
    <run_dir>/trainer_state.pt   # step / episode / best-reward counters
    <run_dir>/rng.pt
    <run_dir>/buffer.pt          # only when saved with save_buffer=True

Training saves this only when the rolling average reward reaches a new best,
so the directory holds the *best* checkpoint, and resuming continues from the
step that checkpoint was taken at.

[read_run_info][phoenx.run_info.read_run_info] is used by ``phoenx-train``
(to check ``--stop_units``) and by the config GUI's Run tab, which calls
``python -m phoenx.run_info DIR`` in a subprocess and reads the JSON it
prints, so the GUI never imports torch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def is_run_dir(path: str | Path) -> bool:
    """True when ``path`` looks like a saved run (has ``config.json`` and ``agent/``)."""
    p = Path(path)
    return (p / "config.json").is_file() and (p / "agent").is_dir()


def read_run_info(run_dir: str | Path) -> dict[str, Any]:
    """Describe a saved run without building it.

    Args:
        run_dir: Directory written by ``Trainer.save``.

    Returns:
        A JSON-serializable dict with ``agent_type``, ``env``, ``stop_unit``,
        ``stop_units``, ``progress`` (steps or episodes reached, in
        ``stop_unit``), ``step``, ``episodes``, ``best_reward``,
        ``has_trainer_state`` and ``has_buffer``. Counter fields are ``None``
        when ``trainer_state.pt`` is missing.

    Raises:
        FileNotFoundError: If ``run_dir`` has no ``config.json``.
    """
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"No config.json in {run_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    schedule = config.get("schedule") or {}
    env = config.get("env") or {}
    env_cfg = env.get("config", env) if isinstance(env, dict) else {}
    info: dict[str, Any] = {
        "run_dir": str(run_dir),
        "agent_type": (config.get("agent") or {}).get("type"),
        "env": env_cfg.get("cfg") if isinstance(env_cfg, dict) else None,
        "env_type": env.get("type") if isinstance(env, dict) else None,
        "buffer_type": (config.get("buffer") or {}).get("type"),
        "stop_unit": schedule.get("stop_unit", "timestep"),
        "stop_units": schedule.get("stop_units"),
        "step": None,
        "episodes": None,
        "best_reward": None,
        "progress": None,
        "has_trainer_state": (run_dir / "trainer_state.pt").is_file(),
        "has_buffer": (run_dir / "buffer.pt").is_file(),
    }
    if info["has_trainer_state"]:
        import torch

        state = torch.load(run_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
        info["step"] = _plain(state.get("_step"))
        info["episodes"] = _plain(state.get("_completed_episodes"))
        info["best_reward"] = _plain(state.get("_best_reward"))
        info["progress"] = info["episodes"] if info["stop_unit"] == "episode" else info["step"]
    return info


def _plain(value: Any) -> Any:
    """Turn tensors / numpy scalars into plain Python numbers."""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            return None
    if isinstance(value, (int, float)):
        return value
    return None


def main(argv: list[str] | None = None) -> int:
    """Print ``read_run_info(DIR)`` as JSON (used by the config GUI)."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m phoenx.run_info RUN_DIR", file=sys.stderr)
        return 2
    try:
        print(json.dumps(read_run_info(args[0])))
    except FileNotFoundError as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
