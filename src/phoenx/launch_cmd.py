"""Build and describe the commands the Run tab launches.

``build_command`` turns form fields into the argument list of a PhoenX
command. ``command_text`` joins that list into one line for this platform.
``describe_run`` writes the one-paragraph summary shown beside a saved
agent folder.
"""

import subprocess
import sys

from phoenx.process_runner import python_command

REPLAY_BUFFERS = ("ReplayBuffer", "PrioritizedReplayBuffer")


def build_command(mode: str, values: dict[str, str]) -> list[str]:
    """Turn Run-tab field values into the argv of a PhoenX command.

    Args:
        mode: ``"train"``, ``"resume"``, ``"test"``, or ``"sweep"``.
        values: Field text, keyed by the Run tab's field names.

    Returns:
        The argv list, starting with the Python interpreter.

    Raises:
        ValueError: The first required field is missing, or ``mode`` is unknown.
    """
    v = {k: (val or "").strip() for k, val in values.items()}

    def need(key: str, label: str) -> str:
        if not v.get(key):
            raise ValueError(f"{label} is required")
        return v[key]

    args: list[str] = []

    def opt(flag: str, key: str) -> None:
        if v.get(key):
            args.extend([flag, v[key]])

    if mode == "train":
        args += ["--config", need("config", "Config file")]
        opt("--stop_units", "stop_units")
        opt("--log_level", "log_level")
        return python_command("phoenx.cli.train", args)
    if mode == "resume":
        args += ["--agent_dir", need("agent_dir", "Agent folder")]
        opt("--stop_units", "stop_units")
        opt("--log_level", "log_level")
        return python_command("phoenx.cli.train", args)
    if mode == "test":
        args += ["--agent_dir", need("agent_dir", "Agent folder")]
        for flag, key in (("--num_episodes", "num_episodes"), ("--num_envs", "num_envs"),
                          ("--render_mode", "render_mode"), ("--env", "env"),
                          ("--seed", "seed"), ("--log_level", "log_level")):
            opt(flag, key)
        return python_command("phoenx.cli.test", args)
    if mode == "sweep":
        args += ["--config", need("sweep", "Sweep file")]
        for flag, key in (("--base-config", "base_config"), ("--num-samples", "num_samples"),
                          ("--max-concurrent", "max_concurrent"), ("--name", "name"),
                          ("--storage-path", "storage_path"), ("--from-phase", "from_phase"),
                          ("--log-level", "log_level")):
            opt(flag, key)
        if v.get("resume") == "1":
            args.append("--resume")
        if v.get("validate_only") == "1":
            args.append("--validate-only")
        return python_command("phoenx.cli.tune", args)
    raise ValueError(f"unknown mode {mode!r}")


def describe_run(info: dict) -> str:
    """Summarize a saved run in one paragraph for the Resume and Test forms.

    Args:
        info: The dict printed by ``python -m phoenx.run_info``, or one with
            an ``error`` key.

    Returns:
        The paragraph shown next to the agent folder.
    """
    if info.get("error"):
        return info["error"]
    what = f"{info.get('agent_type') or '?'} on {info.get('env') or '?'}"
    unit = info.get("stop_unit") or "timestep"
    limit = info.get("stop_units")
    if info.get("progress") is None:
        return f"{what}. No trainer_state.pt, so the step reached is unknown (limit {limit:,} {unit}s)." \
            if isinstance(limit, int) else f"{what}. No trainer_state.pt."
    text = f"{what}. Best checkpoint at {unit} {info['progress']:,}"
    if isinstance(limit, int):
        text += f" of {limit:,}"
    if info.get("best_reward") is not None:
        text += f", best avg reward {info['best_reward']:.2f}"
    text += "."
    if isinstance(limit, int) and info["progress"] >= limit:
        text += " It already reached its limit: set 'Train until' higher to train further."
    if info.get("buffer_type") in REPLAY_BUFFERS and not info.get("has_buffer"):
        text += " No replay buffer was saved, so resuming refills it from scratch."
    return text


def command_text(argv: list[str]) -> str:
    """Join an argv list into the command line the Run tab displays.

    Args:
        argv: The list ``build_command`` returns.

    Returns:
        One command line for the current platform.
    """
    if sys.platform.startswith("win"):
        return subprocess.list2cmdline(argv)
    import shlex
    return shlex.join(argv)
