"""Train / test lifecycle hooks, W&B logging, and Ray Tune reporting.

`Callback` is the no-op base class; subclasses override the hooks they care
about. `Trainer` drives a subset of them at run start, after each env step,
when an episode finishes (the ``*_epoch_end`` hooks — the ``epoch`` argument
is the trainer's global step counter, not an episode index), and at run end.
`WandbCallback` logs metrics and checkpoints to W&B. `RayTuneCallback`
reports training metrics to an enclosing Ray Tune trial on a step cadence and
is a harmless no-op outside a Tune session. Register third-party subclasses
with `register_callback`, look them up with `callback_load`, or rebuild an
instance from a ``get_config()`` dict via `load`.
"""
from pathlib import Path
import math
import os
import json
import warnings
from typing import Optional
import torch as T
import wandb

from . import wandb_support
from .utils import flatten_dict

_CALLBACK_REGISTRY: dict[str, type] = {}

# Bounds the interactive prompt inside wandb.login()'s credential fallback so a
# training run can never block indefinitely waiting on stdin.
_WANDB_LOGIN_TIMEOUT = 30

# Reported for RayTuneCallback's `success_rate` metric when the trainer has no
# `success_criterion` configured (so the key is absent from every episode
# log), keeping the metric key set stable for schedulers/searchers that key
# on it (e.g. ASHA, Optuna).
_DEFAULT_SUCCESS_RATE = 0.0


# Decorator For Third Party Callbacks
def register_callback(name: str | None = None):
    """Decorator that registers a callback class under an explicit name.

    When ``name`` is omitted the class's own ``__name__`` is used, which matches
    the key written by every ``get_config()`` method in this module.

    Args:
        name: Explicit registry key. When omitted, the class's own
            ``__name__`` is used.

    Examples:
        Register under the class name:

            @register_callback()
            class MyCallback(Callback):
                ...

        Register under an explicit alias:

            @register_callback("my_alias")
            class MyCallback(Callback):
                ...
    """
    def decorator(callback_class: type) -> type:
        _CALLBACK_REGISTRY[name or callback_class.__name__] = callback_class
        return callback_class
    return decorator


def callback_load(name: str) -> type:
    """Return the callback class registered under ``name``.

    Args:
        name: Registry key (class ``__name__`` or an alias from
            [register_callback][phoenx.rl_callbacks.register_callback]).

    Returns:
        The registered callback class.

    Raises:
        KeyError: If ``name`` is not in the registry.
    """
    try:
        return _CALLBACK_REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown callback '{name}'. Available: {list(_CALLBACK_REGISTRY)}")



class Callback():
    """No-op base for train / test lifecycle hooks.

    Subclass and override the hooks you need. Concrete subclasses are
    auto-registered under their class name unless ``register=False`` is passed
    to the class statement. `Trainer` currently calls ``on_*_begin`` once at
    run start, ``on_*_step_end`` after every env step, ``on_*_epoch_end`` when
    an episode finishes, and ``on_*_end`` when the loop exits; the ``*_begin``
    step/epoch hooks are reserved for custom loops.
    """

    def __init_subclass__(cls, register: bool = True, **kwargs):
        """Auto-register every concrete subclass by its class name.

        Args:
            register: When ``True`` (default), register the subclass under
                its ``__name__``; when ``False``, skip registration.
            **kwargs (Any): Forwarded to ``object.__init_subclass__``.
        """
        super().__init_subclass__(**kwargs)
        if register:
            _CALLBACK_REGISTRY[cls.__name__] = cls
    
    def on_train_begin(self, logs, models=None):
        """Hook at the start of a training run, before the first env step.

        `Trainer._initialize_run` fires this once when ``context="train"``.
        Use it to open loggers, allocate run state, or inspect the models
        about to be trained.

        Args:
            logs (dict): Trainer config tree (``get_config()``), optionally
                merged with extra kwargs from `_initialize_run`.
            models (list | tuple | None): Agent modules available at run start
                (e.g. policy / critic stacks), or ``None`` to skip watching.
        """
        pass

    def on_train_end(self, logs=None):
        """Hook when the training loop exits.

        `Trainer.train` calls this once per episode log from the final step
        after ``schedule.is_done``. Use it to flush writers or tear down run
        state.

        Args:
            logs (dict | None): Final episode metrics dict, or ``None``.
        """
        pass

    def on_train_epoch_begin(self, epoch: int, logs=None):
        """Reserved hook before a training episode / epoch starts.

        The current `Trainer` does not call this. Override in custom loops
        that mark episode boundaries explicitly.

        Args:
            epoch: Episode or epoch index supplied by the caller.
            logs (dict | None): Optional metrics or context for the new epoch.
        """
        pass

    def on_train_epoch_end(self, epoch: int, logs=None):
        """Hook when a training episode finishes.

        `Trainer.train` calls this for each finished env in a step, passing
        the global step counter as ``epoch`` and the per-episode metrics as
        ``logs``. Typical uses: log episode reward, checkpoint on ``best``.

        Args:
            epoch: Trainer global step at episode end (not an episode index).
            logs (dict | None): Episode metrics (``episode_reward``,
                ``avg_reward``, optional ``best``, …), or ``None``.
        """
        pass

    def on_train_step_begin(self, logs=None):
        """Reserved hook before a training env step.

        The current `Trainer` does not call this. Override in custom loops
        that need pre-step bookkeeping.

        Args:
            logs (dict | None): Optional context for the upcoming step.
        """
        pass

    def on_train_step_end(self, step: int, logs=None):
        """Hook after a training env step (and any learn update that step).

        `Trainer.train` calls this every iteration with the updated global
        step and the step metrics dict (rewards, optional learn metrics).

        Args:
            step: Trainer global step after this env step.
            logs (dict | None): Per-step scalars (e.g. ``step_reward``), or
                ``None``.
        """
        pass

    def on_test_begin(self, logs=None):
        """Hook at the start of an evaluation run, before the first env step.

        `Trainer._initialize_run` fires this once when ``context="test"``.
        Use it to open a separate eval logger or reset eval-only state.

        Args:
            logs (dict | None): Trainer config tree, or ``None``.
        """
        pass

    def on_test_end(self, logs=None):
        """Hook when the evaluation loop exits.

        `Trainer.test` calls this once per episode log from the final step
        after the eval budget is exhausted.

        Args:
            logs (dict | None): Final episode metrics dict, or ``None``.
        """
        pass

    def on_test_epoch_begin(self, epoch: int, logs=None):
        """Reserved hook before an evaluation episode / epoch starts.

        The current `Trainer` does not call this.

        Args:
            epoch: Episode or epoch index supplied by the caller.
            logs (dict | None): Optional metrics or context for the new epoch.
        """
        pass

    def on_test_epoch_end(self, epoch: int, logs=None):
        """Hook when an evaluation episode finishes.

        `Trainer.test` calls this for each finished env, passing the global
        step as ``epoch`` and the per-episode metrics as ``logs``.

        Args:
            epoch: Trainer global step at episode end (not an episode index).
            logs (dict | None): Episode metrics dict, or ``None``.
        """
        pass

    def on_test_step_begin(self, logs=None):
        """Reserved hook before an evaluation env step.

        The current `Trainer` does not call this.

        Args:
            logs (dict | None): Optional context for the upcoming step.
        """
        pass

    def on_test_step_end(self, step: int, logs=None):
        """Hook after an evaluation env step.

        `Trainer.test` calls this every iteration with the updated global
        step and the step metrics dict.

        Args:
            step: Trainer global step after this env step.
            logs (dict | None): Per-step scalars, or ``None``.
        """
        pass

    @classmethod
    def load(cls, config: dict) -> 'Callback':
        """Build an instance by expanding ``config`` as constructor kwargs.

        Args:
            config: Mapping of constructor keyword arguments (not the wrapped
                ``{'type', 'config'}`` form used by
                [load][phoenx.rl_callbacks.load]).

        Returns:
            New instance of this callback class.
        """
        return cls(**config)



class WandbCallback(Callback):
    """Weights & Biases callback that logs metrics and best-model artifacts.

    Authenticates on first train begin, starts a W&B run from the trainer
    config, logs step and episode metrics, and uploads a model artifact when
    an episode log carries ``best=True``. Accepts explicit run naming,
    grouping, tags, and id so every trial of a sweep is distinguishable
    without an extra W&B API call per trial (see ``run_name`` / ``group`` on
    `__init__`).
    """

    def __init__(
        self,
        project_name: str,
        run_name: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        run_id: str | None = None,
        resume: str | bool | None = None,
        sweep_params: dict | None = None,
    ):
        """Store W&B run identity so every trial of a sweep is distinguishable.

        Args:
            project_name: Name of the W&B project.
            run_name: Optional explicit run name, used verbatim as
                ``wandb.init(name=...)``. When given, ``initialize_run`` skips
                the ``wandb_support.get_next_run_number`` API call entirely.
                When omitted, ``initialize_run`` preserves the legacy
                behavior of assigning ``"{prefix}-{run_number}"`` from the
                project's run counter.
            group: Optional explicit W&B run group. When omitted,
                ``initialize_run`` falls back to the legacy
                ``"group-{run_number}"``.
            tags: Optional list of caller-supplied tags. Combined with the
                per-call tags (e.g. ``"train"``/``"test"``) and the agent
                type into a fresh list at run-init time; never mutated.
            run_id: Optional explicit W&B run id, passed through as
                ``wandb.init(id=...)``. Pairs with ``resume`` to reattach to
                a specific run (e.g. one trial's lineage across restores).
            resume: Optional W&B ``resume`` mode (e.g. ``"allow"``, ``True``),
                passed through as ``wandb.init(resume=...)`` when given.
            sweep_params: Optional mapping of swept hyperparameters, flattened
                under a ``sweep/`` prefix and merged into
                ``wandb.init(config=...)`` so W&B's parallel-coordinates and
                parameter-importance panels can chart swept values directly.
        """
        self.project_name = project_name
        self.run_name = run_name
        self.group = group
        self.tags = list(tags) if tags else None
        self.run_id = run_id
        self.resume = resume
        self.sweep_params = sweep_params
        self.save_dir = None
        self.initialized = False
        self._finished = False

    def _ensure_wandb_login(self) -> None:
        """Ensure a W&B session is authenticated before a run starts.

        Resolution order:
            1. If a run is already active (``wandb.run is not None``), do nothing.
            2. The ``WANDB_API_KEY`` environment variable, if set.
            3. A ``wandb_api_key`` file next to this module, if it exists and is
               non-empty; its contents are also written back to
               ``WANDB_API_KEY`` so subsequent calls see it.
            4. If neither explicit source yields a key, fall back to wandb's own
               credential resolution (a cached ``wandb login`` in
               ``~/.netrc``/``~/_netrc``, a settings file, or a bounded
               interactive prompt) via a plain ``wandb.login(relogin=False,
               timeout=_WANDB_LOGIN_TIMEOUT)`` call.

        Raises:
            ValueError: If no credentials can be resolved from any of the
                sources above, i.e. the fallback ``wandb.login`` call returns a
                falsy value or raises (older wandb releases raise
                ``wandb.errors.UsageError`` instead of returning ``False``).
        """
        if wandb.run is not None:
            return
        api_key = os.getenv("WANDB_API_KEY")
        key_path = Path(__file__).with_name("wandb_api_key")
        if not api_key:
            if key_path.exists():
                api_key = key_path.read_text(encoding="utf-8").strip()
                if api_key:
                    os.environ["WANDB_API_KEY"] = api_key

        if api_key:
            wandb.login(key=api_key, relogin=False)
            return

        no_credentials = (
            "WANDB_API_KEY not found and no cached wandb credentials could be "
            "resolved. Set the WANDB_API_KEY environment variable, run "
            f"`wandb login`, or create a key file at {key_path}."
        )

        try:
            logged_in = wandb.login(relogin=False, timeout=_WANDB_LOGIN_TIMEOUT)
        except Exception as exc:
            raise ValueError(no_credentials) from exc

        if not logged_in:
            raise ValueError(no_credentials)

    def initialize_run(self, logs: dict, models: list[T.nn.Module] = None, run_number: Optional[int] = None, run_name_prefix:Optional[str] = None, tags: list[str] = None, job_type: str="train"):
        """Start a W&B run from the trainer config and optionally watch models.

        Reads ``logs['save_dir']`` for artifact paths and calls ``wandb.init``
        with the full config (plus any flattened ``sweep_params``) as
        ``config``. When ``models`` is non-empty, registers each with
        ``wandb.watch``.

        Naming/grouping resolution:
            - ``name``: ``self.run_name`` verbatim when set at construction
              (skipping the ``wandb_support.get_next_run_number`` call
              entirely); otherwise ``"{run_name_prefix}-{run_number}"`` with
              ``run_number`` taken from the ``run_number`` argument or, if
              that is also ``None``, from ``wandb_support.get_next_run_number``
              (today's legacy behavior, preserved exactly).
            - ``group``: ``self.group`` verbatim when set at construction;
              otherwise ``"group-{run_number}"`` using the same
              ``run_number`` resolution as above.
            - ``id`` / ``resume``: passed through only when ``self.run_id`` /
              ``self.resume`` were given at construction.

        Args:
            logs: Trainer config tree; must include ``save_dir`` and
                ``agent['type']``.
            models: Modules to watch, or ``None`` / empty to skip watching.
            run_number: Explicit W&B run counter; when ``None`` and needed for
                naming or grouping, taken from
                ``wandb_support.get_next_run_number``.
            run_name_prefix: Prefix for the legacy run name
                (``"{prefix}-{run_number}"``), used only when no explicit
                ``run_name`` was given at construction.
            tags: Per-call tags (e.g. ``["train"]``); combined with
                ``self.tags`` and the agent type into a fresh list. Never
                mutated.
            job_type: W&B job type string (``"train"`` or ``"test"``).
        """
        # Set save dir
        self.save_dir = logs['save_dir']

        # Resolve the run number lazily and at most once: only needed as a
        # fallback for naming (when no explicit run_name) and/or grouping
        # (when no explicit group), so a fully-named, fully-grouped call
        # (the sweep case) never hits the network.
        resolved_run_number = run_number

        def _resolve_run_number() -> int:
            nonlocal resolved_run_number
            if resolved_run_number is None:
                resolved_run_number = wandb_support.get_next_run_number(self.project_name)
            return resolved_run_number

        if self.run_name:
            name = self.run_name
        else:
            name = f"{run_name_prefix}-{_resolve_run_number()}"

        group = self.group if self.group else f"group-{_resolve_run_number()}"

        combined_tags = list(self.tags) if self.tags else []
        combined_tags.extend(tags or [])
        agent_type = logs.get('agent', {}).get('type')
        if agent_type:
            combined_tags.append(agent_type)

        config = logs
        if self.sweep_params:
            config = {**logs, **flatten_dict(self.sweep_params, parent_key='sweep', sep='/')}

        init_kwargs = dict(
            project=self.project_name,
            name=name,
            tags=combined_tags,
            group=group,
            job_type=job_type,
            config=config,
        )
        if self.run_id:
            init_kwargs['id'] = self.run_id
        if self.resume is not None:
            init_kwargs['resume'] = self.resume

        run = wandb.init(**init_kwargs)
        self.run_name = run.name
        self.initialized = True
        self._finished = False
        if models:
            for i, model in enumerate(models):
                wandb.watch(model, log='all', log_freq=100, idx=i, log_graph=True)

    def on_train_begin(self, logs: dict, run_number: Optional[int] = None, models: Optional[list[T.nn.Module]] = None):
        """Authenticate and start a train W&B run if one is not already open.

        Called by `Trainer._initialize_run` for ``context="train"`` with the
        trainer config, a derived run number, and the agent modules.

        Args:
            logs: Trainer config tree passed to ``wandb.init``.
            run_number: Optional run counter reused when resuming a named run.
            models: Modules to register with ``wandb.watch``, or ``None``.
        """
        self._ensure_wandb_login()
        if not self.initialized:
            self.initialize_run(logs, models, run_number, run_name_prefix="train", tags=["train"], job_type="train")
            

    def _finish_run(self) -> None:
        """Call ``wandb.finish()`` at most once per initialized run.

        `Trainer.train`/`Trainer.test` call ``on_train_end``/``on_test_end``
        once per episode log finished on the loop's final step, so more than
        one env can finish simultaneously and fire this hook multiple times.
        Guards against the resulting double ``wandb.finish()`` call.
        """
        if not self._finished:
            wandb.finish()
            self._finished = True

    def on_train_end(self, logs=None):
        """Finish the active W&B run when training exits.

        Idempotent: safe to call more than once per run.

        Args:
            logs (dict | None): Final episode metrics (unused); accepted for
                base-class compatibility.
        """
        self._finish_run()

    def on_train_epoch_begin(self, epoch: int, logs=None):
        """No-op; W&B logging happens on epoch end.

        Args:
            epoch: Unused epoch / step index.
            logs (dict | None): Unused optional metrics.
        """
        pass

    def on_train_epoch_end(self, epoch: int, logs=None):
        """Log episode metrics to W&B and upload a best-model artifact if flagged.

        Args:
            epoch: Trainer global step used as the W&B ``step``.
            logs (dict | None): Episode metrics; when ``best`` is true, saves
                a model artifact under ``self.save_dir``.
        """
        if logs is None:
            logs = {}
        wandb.log(logs, step=epoch)
        if logs.get("best", False):
            # Create save dir if not exist
            os.makedirs(self.save_dir, exist_ok=True)
            wandb_support.save_model_artifact(self.save_dir, self.project_name, model_is_best=True)

    def on_train_step_begin(self, step: int, logs=None):
        """No-op; W&B logging happens on step end.

        Args:
            step: Unused step index.
            logs (dict | None): Unused optional metrics.
        """
        pass

    def on_train_step_end(self, step: int, logs=None):
        """Log per-step training metrics to W&B.

        Args:
            step: Trainer global step used as the W&B ``step``.
            logs (dict | None): Per-step scalars to log (empty dict if
                ``None``).
        """
        if logs is None:
            logs = {}
        wandb.log(logs, step=step)

    def on_test_begin(self, logs:dict, run_number: Optional[str] = None):
        """Start a test W&B run when none is open yet.

        Called by `Trainer._initialize_run` for ``context="test"``. Skips
        init when a run was already started (e.g. after training).

        Args:
            logs: Trainer config tree passed to ``wandb.init``.
            run_number: Optional run counter / suffix for the test run name.
        """
        if not self.initialized:
            self.initialize_run(logs=logs, run_number=run_number, run_name_prefix="test", tags=["test"], job_type="test")

    def on_test_end(self, logs=None):
        """Finish the active W&B run when evaluation exits.

        Idempotent: safe to call more than once per run.

        Args:
            logs (dict | None): Final episode metrics (unused); accepted for
                base-class compatibility.
        """
        self._finish_run()

    def on_test_epoch_begin(self, epoch: int, logs=None):
        """No-op; W&B logging happens on epoch end.

        Args:
            epoch: Unused epoch / step index.
            logs (dict | None): Unused optional metrics.
        """
        pass

    def on_test_epoch_end(self, epoch: int, logs=None):
        """Log episode evaluation metrics to W&B.

        Args:
            epoch: Trainer global step used as the W&B ``step``.
            logs (dict | None): Episode metrics to log (empty dict if
                ``None``).
        """
        if logs is None:
            logs = {}
        wandb.log(logs, step=epoch)

    def on_test_step_begin(self, step: int, logs=None):
        """No-op; W&B logging happens on step end.

        Args:
            step: Unused step index.
            logs (dict | None): Unused optional metrics.
        """
        pass

    def on_test_step_end(self, step: int, logs=None):
        """Log per-step evaluation metrics to W&B.

        Args:
            step: Trainer global step used as the W&B ``step``.
            logs (dict | None): Per-step scalars to log (empty dict if
                ``None``).
        """
        if logs is None:
            logs = {}
        wandb.log(logs, step=step)

    def get_config(self):
        """Return a serializable ``{'type', 'config'}`` dict for this callback.

        Returns:
            config (dict): Mapping with ``type`` ``"WandbCallback"`` and a
                ``config`` sub-dict of constructor kwargs (``project_name``,
                ``run_name``, ``group``, ``tags``, ``run_id``, ``resume``,
                ``sweep_params``), sufficient to rebuild an equivalent
                callback via `load`.
        """
        return {
            'type': "WandbCallback",
            'config': {
                'project_name': self.project_name,
                'run_name': self.run_name,
                'group': self.group,
                'tags': self.tags,
                'run_id': self.run_id,
                'resume': self.resume,
                'sweep_params': self.sweep_params,
            }
        }

    def save(self, folder: str | Path | None = None):
        """Write this callback's config JSON under ``folder`` or ``save_dir``.

        Args:
            folder: Directory that will receive ``wandb_config.json``, or
                ``None`` to use ``self.save_dir``.
        """
        if folder is None:
            folder = Path(self.save_dir) / "wandb_config.json"
        else:
            folder = Path(folder) / "wandb_config.json"
        os.makedirs(folder.parent, exist_ok=True)
        wandb_config = self.get_config()
        with open(folder, "w", encoding="utf-8") as f:
            json.dump(wandb_config, f)

    @classmethod
    def load(cls, config):
        """Build a `WandbCallback` from a wrapped ``get_config()`` dict.

        Args:
            config (dict): Mapping with a ``'config'`` sub-dict of constructor
                kwargs (as produced by
                [get_config][phoenx.rl_callbacks.WandbCallback.get_config]).

        Returns:
            callback (WandbCallback): New instance from ``config['config']``.
        """
        return cls(**config['config'])


class RayTuneCallback(Callback):
    """Reports training metrics to an enclosing Ray Tune trial on a cadence.

    Outside a Ray Tune session this callback is a harmless no-op (session
    detection is cached after the first check), so it is safe to leave in a
    plain training YAML. Inside a session, it aggregates ``step_log`` values
    between reports (mean over the interval, ignoring non-numeric /
    non-finite values), caches the latest episode metrics, and calls
    ``tune.report`` with a **fixed** key set once the report cadence has
    elapsed: ``timestep``, ``episodes``, ``avg_reward``, ``success_rate``,
    ``episode_reward``, ``episode_steps``, plus the aggregated step/learn
    metric keys observed since the last report. A stable key set matters
    because schedulers/searchers such as ASHA and Optuna error or misbehave
    on a missing metric. Reporting is suppressed until the first episode
    completes, so no searcher ever sees ``-inf``/``nan`` for its metric.

    ``from ray import tune`` is imported lazily inside methods so ordinary
    (non-Tune) training runs never pay Ray's import cost, and there is no
    module-level ``ray`` import.

    Examples:
        Report every 50,000 timesteps to the enclosing Tune trial::

            from phoenx.rl_callbacks import RayTuneCallback

            callback = RayTuneCallback(every=50000, unit="timestep")
            # trainer = Trainer.load(config, callbacks=[callback]) / via YAML
            trainer.train()
    """

    def __init__(self, every: int = 50000, unit: str = "timestep"):
        """Set the report cadence.

        Args:
            every: Report interval, in ``unit`` units. Matches the sweep
                YAML's ``report: {every: ..., unit: ...}`` block.
            unit: Cadence unit, either ``"timestep"`` (Trainer global step,
                as seen by `on_train_step_end`) or ``"episode"`` (cumulative
                completed-episode count, as seen by `on_train_epoch_end`).

        Raises:
            ValueError: If ``unit`` is not ``"timestep"`` or ``"episode"``.
        """
        if unit not in ("timestep", "episode"):
            raise ValueError(
                f"Unknown RayTuneCallback unit '{unit}'. Expected 'timestep' or 'episode'."
            )
        self.every = every
        self.unit = unit

        self._trainer = None
        self._session_active: Optional[bool] = None

        self._episode_completed = False
        self._episodes = 0
        self._avg_reward = None
        self._success_rate = _DEFAULT_SUCCESS_RATE
        self._episode_reward = None
        self._episode_steps = None

        self._step_sums: dict[str, float] = {}
        self._step_counts: dict[str, int] = {}

        self._last_report_timestep = 0
        self._last_report_episode = 0

    def bind(self, trainer) -> None:
        """Store a reference to the owning trainer.

        This is the checkpointing seam for a later PBT (Population Based
        Training) pass: a scheduler that mutates hyperparameters mid-run
        needs the callback to be able to trigger
        ``trainer.save(...)``/``tune.report(..., checkpoint=...)`` on the
        checkpoint cadence. Pass 1 only stores the reference; no
        checkpointing is implemented here.

        Args:
            trainer (phoenx.trainer.Trainer): The `phoenx.trainer.Trainer`
                (or compatible object exposing ``save``) driving this
                callback.
        """
        self._trainer = trainer

    def _detect_session(self) -> bool:
        """Detect (and cache) whether a Ray Tune trial is currently active.

        Uses ``tune.get_context().get_trial_id() is not None``, which Ray
        confirms returns ``None`` outside a Tune session while emitting a
        ``UserWarning``; that warning is suppressed here since a ``None``
        result is an expected, harmless outcome for this callback.

        Returns:
            ``True`` when running inside a Ray Tune trial, ``False``
            otherwise. Cached after the first call.
        """
        if self._session_active is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                from ray import tune
                self._session_active = tune.get_context().get_trial_id() is not None
        return self._session_active

    def _accumulate(self, logs: dict) -> None:
        """Fold finite numeric values from a step log into the running mean.

        Args:
            logs: Per-step metrics dict (e.g. ``step_log``). Non-numeric
                values (``bool`` included) and non-finite floats
                (``nan``/``inf``) are skipped so they cannot poison the
                aggregate.
        """
        for key, value in logs.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(value):
                continue
            self._step_sums[key] = self._step_sums.get(key, 0.0) + float(value)
            self._step_counts[key] = self._step_counts.get(key, 0) + 1

    def _cadence_elapsed(self, step: int) -> bool:
        """Check whether the configured report interval has elapsed.

        Args:
            step: Current trainer global step (as passed to
                `on_train_step_end`).

        Returns:
            ``True`` when ``every`` units (timesteps or episodes, per
            ``self.unit``) have passed since the last report.
        """
        if self.unit == "timestep":
            return step - self._last_report_timestep >= self.every
        return self._episodes - self._last_report_episode >= self.every

    def _report(self, step: int) -> None:
        """Aggregate, report to Ray Tune, and reset the aggregation buffers.

        Args:
            step: Current trainer global step, reported as ``timestep``.
        """
        from ray import tune

        metrics = {
            key: self._step_sums[key] / self._step_counts[key]
            for key in self._step_sums
        }
        metrics.update({
            "timestep": step,
            "episodes": self._episodes,
            "avg_reward": self._avg_reward,
            "success_rate": self._success_rate,
            "episode_reward": self._episode_reward,
            "episode_steps": self._episode_steps,
        })
        tune.report(metrics)

        self._step_sums = {}
        self._step_counts = {}
        self._last_report_timestep = step
        self._last_report_episode = self._episodes

    def on_train_begin(self, logs, models=None):
        """Detect (and cache) Ray Tune session status at run start.

        Args:
            logs (dict): Trainer config tree (unused beyond triggering
                session detection); accepted for base-class compatibility.
            models (list | tuple | None): Unused; accepted for base-class
                compatibility.
        """
        self._detect_session()

    def on_train_epoch_end(self, epoch: int, logs=None):
        """Cache the latest episode metrics for the next report.

        Args:
            epoch: Trainer global step at episode end (unused; caching is
                keyed off the episode log itself).
            logs (dict | None): Episode metrics dict (``avg_reward``,
                ``episode_reward``, ``episode_steps``, cumulative
                ``episode`` count, optional ``success_rate``). Marks the
                first-episode-completed gate once any log arrives.
        """
        if logs is None:
            return
        self._episode_completed = True
        if "avg_reward" in logs:
            self._avg_reward = logs["avg_reward"]
        if "episode_reward" in logs:
            self._episode_reward = logs["episode_reward"]
        if "episode_steps" in logs:
            self._episode_steps = logs["episode_steps"]
        if "success_rate" in logs:
            self._success_rate = logs["success_rate"]
        if "episode" in logs:
            self._episodes = logs["episode"]

    def on_train_step_end(self, step: int, logs=None):
        """Aggregate step metrics and report to Ray Tune on cadence.

        No-op outside a Ray Tune session (detected lazily here if
        `on_train_begin` was never called) and before the first episode has
        completed, so a searcher never observes a missing-data metric.

        Args:
            step: Trainer global step after this env step; reported as
                ``timestep``.
            logs (dict | None): Per-step scalars (rewards, and on a learn
                iteration, algorithm-specific learn metrics), aggregated by
                mean since the last report.
        """
        if not self._detect_session():
            return
        if logs:
            self._accumulate(logs)
        if not self._episode_completed:
            return
        if self._cadence_elapsed(step):
            self._report(step)

    def get_config(self) -> dict:
        """Return a serializable ``{'type', 'config'}`` dict for this callback.

        Returns:
            config (dict): Mapping with ``type`` ``"RayTuneCallback"`` and a
                ``config`` sub-dict of constructor kwargs (``every``,
                ``unit``), sufficient to rebuild an equivalent callback via
                `load`.
        """
        return {
            "type": "RayTuneCallback",
            "config": {
                "every": self.every,
                "unit": self.unit,
            },
        }

    @classmethod
    def load(cls, config: dict) -> "RayTuneCallback":
        """Build a `RayTuneCallback` from a wrapped ``get_config()`` dict.

        Args:
            config: Mapping with a ``'config'`` sub-dict of constructor
                kwargs (as produced by
                [get_config][phoenx.rl_callbacks.RayTuneCallback.get_config]).

        Returns:
            callback (RayTuneCallback): New instance from ``config['config']``.
        """
        return cls(**config['config'])


def load(config: dict):
    """Instantiate a callback from a config dict produced by ``get_config()``.

    Args:
        config: Must contain a ``'type'`` key whose value matches a
            registered callback class name, plus a ``'config'`` sub-dict of
            constructor kwargs.

    Returns:
        callback (Callback): An instance of the requested callback class.
    """
    cb_type = config.get("type", "")
    if cb_type not in _CALLBACK_REGISTRY:
        raise ValueError(
            f"Unknown callback type '{cb_type}'. "
            f"Available: {list(_CALLBACK_REGISTRY)}"
        )
    return _CALLBACK_REGISTRY[cb_type].load(config)
