"""The structure of a sweep config, without importing Ray or torch.

``phoenx.ray_tune`` imports these key sets to reject misspelled keys before a
sweep starts (a misspelled phase key such as ``serach_space`` used to be
ignored, so that phase ran with nothing to search). The config GUI's sweep
editor uses the same definitions.
"""

from __future__ import annotations

#: Keys a sweep file may have at the top level. The last four may also appear
#: at the top level instead of under ``phases`` (an implicit single phase).
SWEEP_KEYS: dict[str, str] = {
    "base_config": "Training config the sweep starts from: a bundled example name or a path.",
    "overrides": "Dotted config path -> constant, applied to every trial of every phase.",
    "defaults": "Phase settings inherited by every phase; a phase's own value wins.",
    "blocks": "Reusable layer templates for architecture search, by name.",
    "phases": "Phases run in order; each can promote its best config into the next.",
    "ray_init": "Keyword arguments forwarded to ray.init(), e.g. runtime_env.",
    "search_space": "Single-phase shorthand: dotted config path -> search spec.",
    "tune": "Single-phase shorthand: Ray Tune settings.",
    "architecture": "Single-phase shorthand: per-module depth / block search.",
    "optimizers": "Single-phase shorthand: per-module optimizer search.",
}

#: Keys a phase (or ``defaults``) may have.
PHASE_KEYS: dict[str, str] = {
    "name": "Phase name (unique); results are stored under it.",
    "search_space": "Dotted config path -> search spec, e.g. "
                    "agent.config.discount: {dist: uniform, low: 0.95, high: 0.999}.",
    "architecture": "Module (branches.policy, trunk, ...) -> {depth, blocks} to search layer stacks.",
    "optimizers": "Module -> optimizer fields to search, e.g. {lr: {dist: loguniform, ...}}.",
    "tune": "Ray Tune settings: num_samples, search_alg, scheduler, stop, ...",
    "promote": "What to carry into the next phase: {mode: best} promotes the best trial's config.",
    "metric": "Metric the search optimizes, e.g. avg_reward.",
    "mode": "max or min: whether a higher or lower metric is better.",
    "resources": "Resources per trial, e.g. {cpu: 1, gpu: 0.25}.",
    "max_concurrent_trials": "How many trials run at once.",
    "report": "How often trials report the metric, e.g. {every: 20000, unit: timestep}.",
    "auto_learn_every": "Derive schedule.learn_every from the trial's buffer / env sizes.",
    "wandb": "Log trials to Weights & Biases, e.g. {project: my-sweep}.",
}

#: Keys of a phase's ``tune`` block. ``search_alg``, ``scheduler`` and
#: ``stop`` contents are passed to Ray and are not checked here.
TUNE_KEYS: dict[str, str] = {
    "num_samples": "Number of trials to run in this phase.",
    "search_alg": "Search algorithm, e.g. {type: random} or {type: optuna}.",
    "scheduler": "Trial scheduler, e.g. {type: asha, time_attr: timestep, grace_period: ..., max_t: ...}.",
    "stop": "Stopping rules, e.g. [{type: TrialPlateauStopper, metric: avg_reward, std: 5.0}].",
    "time_budget_s": "Stop the phase after this many seconds.",
    "reuse_actors": "Reuse Ray actors between trials (faster startup).",
}

#: Search-spec distributions and their keys: (required, optional).
DISTRIBUTIONS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "uniform": (("low", "high"), ()),
    "loguniform": (("low", "high"), ()),
    "quniform": (("low", "high", "q"), ()),
    "randint": (("lower", "upper"), ()),
    "qrandint": (("lower", "upper", "q"), ()),
    "lograndint": (("lower", "upper"), ("base",)),
    "choice": (("values",), ()),
    "grid_search": (("values",), ()),
    "randn": ((), ("mean", "sd")),
    "fixed": (("value",), ()),
}


def check_sweep_keys(sweep: dict) -> None:
    """Reject misspelled keys at the sweep, phase and ``tune`` levels.

    Keys starting with ``_`` are internal and always allowed.

    Args:
        sweep: Parsed sweep config.

    Raises:
        ValueError: Naming each unknown key with a "did you mean" suggestion.
    """
    from phoenx.config_utils import reject_unknown_keys

    def check(owner: str, mapping, valid) -> None:
        if not isinstance(mapping, dict):
            return
        try:
            reject_unknown_keys(owner, list(mapping), valid)
        except TypeError as exc:
            raise ValueError(str(exc)) from None

    check("Sweep config", sweep, SWEEP_KEYS)
    check("sweep['defaults']", sweep.get("defaults"), PHASE_KEYS)
    phase_like = [("top-level phase", sweep)] if "phases" not in sweep else []
    for i, phase in enumerate(sweep.get("phases") or []):
        if isinstance(phase, dict):
            label = f"Phase '{phase.get('name', i)}'"
            check(label, phase, PHASE_KEYS)
            phase_like.append((label, phase))
    for label, phase in phase_like + [("sweep['defaults']", sweep.get("defaults") or {})]:
        if isinstance(phase, dict):
            check(f"{label} tune", phase.get("tune"), TUNE_KEYS)
