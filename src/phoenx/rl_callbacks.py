"""Train / test lifecycle hooks, W&B and MLflow logging, and Ray Tune reporting.

`Callback` is the no-op base class; subclasses override the hooks they care
about. `Trainer` drives a subset of them at run start, after each env step,
when an episode finishes (the ``*_epoch_end`` hooks — the ``epoch`` argument
is the trainer's global step counter, not an episode index), and at run end.
`WandbCallback` logs metrics and checkpoints to W&B; `MLflowCallback` does the
same for MLflow (an optional dependency, imported only when used). `RayTuneCallback`
reports training metrics to an enclosing Ray Tune trial on a step cadence and
is a harmless no-op outside a Tune session. Register third-party subclasses
with `register_callback`, look them up with `callback_load`, or rebuild an
instance from a ``get_config()`` dict via `load`.
"""
from pathlib import Path
import math
import os
import atexit
import json
import time
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

# Bounds MLflowCallback's tracking-server health check so a stopped server
# fails the run fast instead of stalling in MLflow's own retry loop.
_MLFLOW_CONNECT_TIMEOUT = 5

# MLflow storage limits: param values are capped (6000 chars in MLflow 2.x/3.x)
# and keys at 250 chars.
_MLFLOW_MAX_PARAM_LEN = 6000
_MLFLOW_MAX_KEY_LEN = 250

# MLflowCallback buffers metrics and sends them in batches (like W&B's client
# does) instead of one HTTP request per env step: a batch goes out once it
# holds _MLFLOW_BATCH_SIZE values (MLflow's per-request limit) or the oldest
# value is _MLFLOW_FLUSH_SECONDS old, and always at run end.
_MLFLOW_BATCH_SIZE = 1000
_MLFLOW_FLUSH_SECONDS = 5.0

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



def _test_run_name(run_name: str | None) -> str | None:
    """Name for a test run of an agent whose saved config carries ``run_name``.

    A saved config keeps the *training* run's name, so a later test of that
    agent would otherwise reuse it. The test run is named after it instead:
    ``train-3`` -> ``test-3`` (same number, so the pair lines up and shares
    ``group-3``), any other name gets a ``test-`` prefix
    (``mysweep-refine-0004of0008`` -> ``test-mysweep-refine-0004of0008``),
    and a name that is already a test name is kept. ``None`` stays ``None``
    so the callback picks the next free ``test-N``.
    """
    if not run_name:
        return None
    if run_name.startswith("test-"):
        return run_name
    if run_name.startswith("train-"):
        return "test-" + run_name[len("train-"):]
    return f"test-{run_name}"


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
        ``logs``. Typical uses: log episode reward, checkpoint on ``saved``.

        Args:
            epoch: Trainer global step at episode end (not an episode index).
            logs (dict | None): Episode metrics (``episode_reward``,
                ``avg_reward``, optional ``best``, optional ``saved``, …), or
                ``None``. ``best`` marks a new rolling-average best reward;
                ``saved`` marks that the trainer actually wrote a checkpoint
                to disk this step (gated by ``schedule.save_every``).
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
    an episode log carries ``saved=True`` (i.e. the trainer actually wrote a
    checkpoint to disk this step), subject to its own optional additional
    cooldown (``artifact_every``). Accepts explicit run naming, grouping,
    tags, and id so every trial of a sweep is distinguishable without an
    extra W&B API call per trial (see ``run_name`` / ``group`` on
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
        artifact_every: int | None = None,
        extra_config: dict | None = None,
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
            artifact_every: Minimum number of trainer timesteps between
                artifact uploads, applied on top of the trainer's own
                ``schedule.save_every`` checkpoint cadence. ``None`` (the
                default) or ``0`` means no additional cooldown: an artifact
                is uploaded on every step the trainer actually wrote a
                checkpoint (``logs['saved']``).
            extra_config: Optional mapping already flat (no further
                flattening applied), merged into ``wandb.init(config=...)``
                last — after ``logs`` and any flattened ``sweep_params`` — so
                it wins any key collision. Exists for run provenance that is
                not itself a swept hyperparameter (e.g. the underlying Ray
                Tune trial id), and so is expected to stay out of the
                ``sweep/`` namespace that W&B's parallel-coordinates and
                parameter-importance panels read. Produced by
                `_inject_trial_callbacks` in `phoenx.ray_tune`; a
                hand-written training YAML has no reason to set it.
        """
        self.project_name = project_name
        self.run_name = run_name
        self.group = group
        self.tags = list(tags) if tags else None
        self.run_id = run_id
        self.resume = resume
        self.sweep_params = sweep_params
        self.artifact_every = artifact_every
        self.extra_config = dict(extra_config) if extra_config else None
        self.save_dir = None
        self.initialized = False
        self._finished = False
        self._last_artifact_at = None

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
        with the full config (plus any flattened ``sweep_params``, then
        ``self.extra_config`` merged last so it wins any key collision) as
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
        if self.extra_config:
            config = {**config, **self.extra_config}

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

    def _should_upload_artifact(self, epoch: int) -> bool:
        """Return whether the artifact upload's own cooldown has elapsed.

        Args:
            epoch: Current trainer global step (as passed to
                `on_train_epoch_end`).

        Returns:
            ``True`` when ``artifact_every`` is ``None``/``0`` (no
                additional cooldown beyond the trainer's own save cadence),
                or when no artifact has been uploaded yet this run, or when
                ``epoch`` has advanced by at least ``artifact_every`` steps
                since the last upload.
        """
        if not self.artifact_every:
            return True
        if self._last_artifact_at is None:
            return True
        return epoch >= self._last_artifact_at + self.artifact_every

    def on_train_epoch_end(self, epoch: int, logs=None):
        """Log episode metrics to W&B and upload a best-model artifact if flagged.

        Uploads follow ``logs['saved']`` — set by the trainer only when it
        actually wrote a checkpoint to disk this step — rather than
        ``logs['best']``, so an upload never ships a stale model when the
        trainer's own save cooldown skipped the write. ``artifact_every``
        can throttle uploads further on top of that.

        Args:
            epoch: Trainer global step used as the W&B ``step`` and as the
                cooldown clock for ``artifact_every``.
            logs (dict | None): Episode metrics; when ``saved`` is true and
                the upload cooldown has elapsed, saves a model artifact
                under ``self.save_dir``.
        """
        if logs is None:
            logs = {}
        wandb.log(logs, step=epoch)
        if logs.get("saved", False) and self._should_upload_artifact(epoch):
            # Create save dir if not exist
            os.makedirs(self.save_dir, exist_ok=True)
            wandb_support.save_model_artifact(self.save_dir, self.project_name, model_is_best=True)
            self._last_artifact_at = epoch

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
        init when a run was already started (e.g. after training in the same
        process). Otherwise the run name inherited from the agent's saved
        training config is turned into a test name (see `_test_run_name`).

        Args:
            logs: Trainer config tree passed to ``wandb.init``.
            run_number: Optional run counter / suffix for the test run name.
        """
        if not self.initialized:
            self.run_name = _test_run_name(self.run_name)
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
                ``sweep_params``, ``artifact_every``, ``extra_config``),
                sufficient to rebuild an equivalent callback via `load`.
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
                'artifact_every': self.artifact_every,
                'extra_config': self.extra_config,
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


def _import_mlflow():
    """Import ``mlflow`` lazily so it stays an optional dependency.

    Returns:
        The imported ``mlflow`` module.

    Raises:
        ImportError: If MLflow is not installed, with the install command.
    """
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "MLflowCallback needs MLflow. Install it with `pip install mlflow` "
            "(or `pip install phoenx[mlflow]`)."
        ) from exc
    return mlflow


def _mlflow_key(key) -> str:
    """Return ``key`` with characters MLflow rejects replaced by ``_``.

    MLflow metric/param/tag names allow letters, digits, ``_ - . / :`` and
    spaces; PhoenX metric names (``rollout/truncation_rate``) already fit.
    """
    return "".join(c if (c.isalnum() or c in "_-./: ") else "_" for c in str(key))


def _mlflow_scalar(value) -> float | None:
    """Coerce a log value to ``float``, or ``None`` when it is not a scalar.

    W&B accepts any object; MLflow metrics must be numbers. Bools (``saved``,
    ``best``) become ``0.0``/``1.0``; numpy scalars and one-element tensors
    are unwrapped; strings, lists and other objects are skipped.
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, T.Tensor):
        return float(value.item()) if value.numel() == 1 else None
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", None) in ((), (1,)):
        try:
            return float(item())
        except (TypeError, ValueError):
            return None
    return None


class MLflowCallback(Callback):
    """MLflow callback that logs metrics and best-model artifacts.

    Drop-in counterpart of `WandbCallback`: same constructor arguments, same
    hooks, same artifact rules. ``project_name`` is the MLflow *experiment*
    name. On first train begin it makes sure the tracking server is up
    (starting a local one if needed, see `phoenx.mlflow_server`), starts an
    MLflow run from the trainer config, logs step and episode metrics, and
    uploads the save directory as the ``model`` artifact when an episode log
    carries ``saved=True``, subject to the optional ``artifact_every``
    cooldown.

    Where MLflow differs from W&B:

    - ``group`` and the caller's ``tags`` become run tags (MLflow has no run
      groups); ``job_type`` and the agent type are tags too.
    - The config is flattened to ``/``-separated params (MLflow params are
      flat strings) and the full tree is also stored as ``config.json``.
    - Only numeric log values become metrics; bools log as ``0``/``1`` and
      anything else is skipped.
    - There is no ``wandb.watch``: when models are passed, their
      architecture and parameter counts are logged instead of histograms.
    - Each upload replaces the run's ``model`` artifact folder, so the run
      holds the latest best checkpoint (W&B keeps every version).
    - ``run_id`` must be an existing MLflow run id (MLflow assigns ids).
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
        artifact_every: int | None = None,
        extra_config: dict | None = None,
        tracking_uri: str | None = None,
        async_logging: bool = True,
        auto_start_server: bool = True,
    ):
        """Store MLflow run identity so every trial of a sweep is distinguishable.

        Args:
            project_name: MLflow experiment name; created if missing.
            run_name: Optional explicit run name. When omitted,
                ``initialize_run`` names the run ``"{prefix}-{run_number}"``
                from the experiment's existing run names, like W&B.
            group: Optional group label, stored as the ``group`` tag. When
                omitted, ``"group-{run_number}"`` is used.
            tags: Optional caller tags; each becomes a tag set to ``"true"``,
                combined with the per-call tags and agent type. Never mutated.
            run_id: Optional existing MLflow run id to reattach to.
            resume: W&B-style resume mode. With ``run_id`` set, ``"must"``
                raises if the run does not exist; any other value starts a
                new run in that case (with a warning).
            sweep_params: Optional swept hyperparameters, logged as params
                under a ``sweep/`` prefix.
            artifact_every: Minimum trainer timesteps between artifact
                uploads on top of the trainer's own save cadence. ``None``
                or ``0`` uploads on every step the trainer wrote a checkpoint.
            extra_config: Optional flat mapping merged into the logged config
                last (e.g. Ray trial provenance), so it wins key collisions.
            tracking_uri: MLflow tracking URI. When omitted,
                ``MLFLOW_TRACKING_URI`` is used, then
                ``http://127.0.0.1:5000`` (see `phoenx.mlflow_server`).
            async_logging: Send metric batches from a background thread so
                logging never blocks the training loop. Flushed when the run
                ends.
            auto_start_server: When the tracking URI is on this machine
                (``127.0.0.1`` / ``localhost``) and no server answers, start
                one in the background (`phoenx.mlflow_server.ensure_server`).
                It keeps running after the run so the dashboard stays up.

        Raises:
            ImportError: If MLflow is not installed.
        """
        _import_mlflow()
        self.project_name = project_name
        self.run_name = run_name
        self.group = group
        self.tags = list(tags) if tags else None
        self.run_id = run_id
        self.resume = resume
        self.sweep_params = sweep_params
        self.artifact_every = artifact_every
        self.extra_config = dict(extra_config) if extra_config else None
        self.tracking_uri = tracking_uri
        self.async_logging = async_logging
        self.auto_start_server = auto_start_server
        self.save_dir = None
        self.initialized = False
        self._finished = False
        self._last_artifact_at = None
        self._client = None
        self._skipped_keys: set[str] = set()
        self._pending: list = []
        self._last_flush = 0.0
        # Filled once a run starts; written by `save` for links (e.g. the GUI).
        self.experiment_id = None
        self.active_run_id = None
        self.resolved_tracking_uri = None

    # ------------------------------------------------------------------ setup
    def _ensure_mlflow_tracking(self) -> None:
        """Resolve the tracking URI and make sure its server is reachable.

        Resolution order:
            1. If a run is already open on this callback, do nothing.
            2. ``tracking_uri`` from the constructor.
            3. The ``MLFLOW_TRACKING_URI`` environment variable.
            4. ``phoenx.mlflow_server.DEFAULT_TRACKING_URI``
               (``http://127.0.0.1:5000``).

        For a local ``http(s)`` URI with ``auto_start_server`` on, a server
        is started if none answers (`phoenx.mlflow_server.ensure_server`).
        Otherwise the ``/health`` endpoint is checked with a
        ``_MLFLOW_CONNECT_TIMEOUT`` second timeout, so a missing server fails
        the run fast instead of stalling in MLflow's long retry loop.

        Raises:
            ValueError: If the server cannot be reached or started.
        """
        if self._client is not None and self.initialized and not self._finished:
            return
        from . import mlflow_server
        mlflow = _import_mlflow()
        uri = self.tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or mlflow_server.DEFAULT_TRACKING_URI
        mlflow.set_tracking_uri(uri)

        if uri.startswith(("http://", "https://")):
            if self.auto_start_server and mlflow_server.is_local(uri):
                mlflow_server.ensure_server(uri)
            elif not mlflow_server.is_up(uri, timeout=_MLFLOW_CONNECT_TIMEOUT):
                raise ValueError(
                    f"MLflow tracking server at {uri} is not reachable. Start it "
                    "with `python -m phoenx.mlflow_server start`, or set "
                    "MLFLOW_TRACKING_URI / tracking_uri to a running server."
                )

        self.resolved_tracking_uri = uri
        self._client = mlflow.MlflowClient(tracking_uri=uri)

    def _get_experiment_id(self) -> str:
        """Return the id of experiment ``project_name``, creating it if needed.

        Safe when several sweep trials start at once: if another process
        creates the experiment between the lookup and the create, the
        create fails and the lookup is retried.
        """
        experiment = self._client.get_experiment_by_name(self.project_name)
        if experiment is None:
            try:
                return self._client.create_experiment(self.project_name)
            except _import_mlflow().exceptions.MlflowException:
                experiment = self._client.get_experiment_by_name(self.project_name)
                if experiment is None:
                    raise
        if experiment.lifecycle_stage == "deleted":
            self._client.restore_experiment(experiment.experiment_id)
        return experiment.experiment_id

    def _next_run_number(self) -> int:
        """Return one more than the highest ``*-N`` run name in the experiment.

        The MLflow version of ``wandb_support.get_next_run_number``. Names
        that do not end in ``-<int>`` are ignored rather than crashing.
        """
        highest = 0
        page_token = None
        while True:
            runs = self._client.search_runs(
                [self.experiment_id], max_results=1000, page_token=page_token
            )
            for run in runs:
                suffix = (run.info.run_name or "").rsplit("-", 1)[-1]
                if suffix.isdigit():
                    highest = max(highest, int(suffix))
            page_token = getattr(runs, "token", None)
            if not page_token:
                return highest + 1

    def initialize_run(self, logs: dict, models: list[T.nn.Module] = None, run_number: Optional[int] = None, run_name_prefix: Optional[str] = None, tags: list[str] = None, job_type: str = "train"):
        """Start an MLflow run from the trainer config.

        Naming, grouping and config composition match
        `WandbCallback.initialize_run`: ``self.run_name`` verbatim when set,
        else ``"{run_name_prefix}-{run_number}"``; ``self.group`` when set,
        else ``"group-{run_number}"``; config is ``logs`` plus flattened
        ``sweep_params`` plus ``extra_config`` (last wins). The run number is
        looked up only when needed.

        Args:
            logs: Trainer config tree; must include ``save_dir``.
            models: Modules whose architecture is logged, or ``None``.
            run_number: Explicit run counter for naming/grouping.
            run_name_prefix: Prefix for the default run name.
            tags: Per-call tags (e.g. ``["train"]``). Never mutated.
            job_type: ``"train"`` or ``"test"``, stored as a tag.

        Raises:
            ValueError: If ``resume="must"`` and ``run_id`` does not exist.
        """
        self.save_dir = logs['save_dir']
        self.experiment_id = self._get_experiment_id()

        resolved_run_number = run_number

        def _resolve_run_number():
            nonlocal resolved_run_number
            if resolved_run_number is None:
                resolved_run_number = self._next_run_number()
            return resolved_run_number

        name = self.run_name if self.run_name else f"{run_name_prefix}-{_resolve_run_number()}"
        group = self.group if self.group else f"group-{_resolve_run_number()}"

        combined_tags = list(self.tags) if self.tags else []
        combined_tags.extend(tags or [])
        agent_type = logs.get('agent', {}).get('type')
        if agent_type:
            combined_tags.append(agent_type)
        run_tags = {_mlflow_key(t): "true" for t in combined_tags}
        run_tags.update({
            "group": str(group),
            "job_type": job_type,
            "phoenx.tags": ",".join(str(t) for t in combined_tags),
        })
        if agent_type:
            run_tags["agent_type"] = str(agent_type)

        config = logs
        if self.sweep_params:
            config = {**logs, **flatten_dict(self.sweep_params, parent_key='sweep', sep='/')}
        if self.extra_config:
            config = {**config, **self.extra_config}

        run = self._open_run(name, run_tags)
        self.active_run_id = run.info.run_id
        self.run_name = run.info.run_name
        self.initialized = True
        self._finished = False
        self._skipped_keys = set()
        self._pending = []
        self._last_flush = time.monotonic()
        # Safety net for runs that never reach on_*_end (Ctrl+C, a crash, the
        # GUI's Stop): flush and mark the run KILLED at interpreter exit,
        # the way wandb closes its own runs.
        atexit.register(self._close_at_exit)

        self._log_config(config)
        if models:
            self._log_models(models)

    def _open_run(self, name: str, run_tags: dict):
        """Reattach to ``self.run_id`` when it exists, else create a new run."""
        mlflow = _import_mlflow()
        if self.run_id:
            try:
                run = self._client.get_run(self.run_id)
            except mlflow.exceptions.MlflowException:
                run = None
            if run is not None:
                self._client.update_run(run.info.run_id, status="RUNNING")
                self._client.log_batch(run.info.run_id, tags=self._tag_list(run_tags))
                return self._client.get_run(run.info.run_id)
            if self.resume == "must":
                raise ValueError(f"MLflow run '{self.run_id}' does not exist and resume='must'.")
            warnings.warn(f"MLflow run '{self.run_id}' not found; starting a new run instead.")
        return self._client.create_run(self.experiment_id, run_name=name, tags=run_tags)

    @staticmethod
    def _tag_list(tags: dict) -> list:
        """Convert a tag dict to MLflow ``RunTag`` entities."""
        from mlflow.entities import RunTag
        return [RunTag(k, v) for k, v in tags.items()]

    def _log_config(self, config: dict) -> None:
        """Log ``config`` as flat params plus the full tree as ``config.json``.

        Params are capped at MLflow's value length and sent in batches of
        100. On a resumed run a param whose value changed cannot be
        overwritten; that batch is skipped with a warning.
        """
        from mlflow.entities import Param
        flat = flatten_dict(config, sep='/')
        params = [
            Param(_mlflow_key(k)[:_MLFLOW_MAX_KEY_LEN], str(v)[:_MLFLOW_MAX_PARAM_LEN])
            for k, v in flat.items()
        ]
        mlflow = _import_mlflow()
        for i in range(0, len(params), 100):
            try:
                self._client.log_batch(self.active_run_id, params=params[i:i + 100])
            except mlflow.exceptions.MlflowException as exc:
                warnings.warn(f"MLflowCallback: some params were not logged: {exc}")
        self._client.log_text(
            self.active_run_id, json.dumps(config, indent=2, default=str), "config.json"
        )

    def _log_models(self, models) -> None:
        """Log each model's architecture and parameter counts (no ``watch`` in MLflow).

        Writes ``model_architecture/model_<i>.txt`` (the module tree as
        printed by PyTorch) and params ``model_<i>/class``,
        ``model_<i>/num_params`` and ``model_<i>/num_trainable_params``.
        Objects that are not ``nn.Module`` get the text file and class only.
        """
        from mlflow.entities import Param
        params = []
        for i, model in enumerate(models):
            self._client.log_text(self.active_run_id, str(model), f"model_architecture/model_{i}.txt")
            params.append(Param(f"model_{i}/class", type(model).__name__))
            if not isinstance(model, T.nn.Module):
                continue  # e.g. an intrinsic-motivation wrapper: text only
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            params += [Param(f"model_{i}/num_params", str(total)),
                       Param(f"model_{i}/num_trainable_params", str(trainable))]
        try:
            self._client.log_batch(self.active_run_id, params=params)
        except _import_mlflow().exceptions.MlflowException as exc:
            warnings.warn(f"MLflowCallback: model params were not logged: {exc}")

    # ---------------------------------------------------------------- logging
    def _log(self, logs: dict | None, step: int) -> None:
        """Log every numeric value in ``logs`` as a metric at ``step``.

        The MLflow version of ``wandb.log(logs, step=step)``. Values are
        buffered and sent by `_flush` (see ``_MLFLOW_BATCH_SIZE`` /
        ``_MLFLOW_FLUSH_SECONDS``). Non-numeric values are skipped, with one
        warning per key per run.
        """
        if not logs:
            return
        from mlflow.entities import Metric
        timestamp = int(time.time() * 1000)
        for key, value in logs.items():
            number = _mlflow_scalar(value)
            if number is None:
                if key not in self._skipped_keys:
                    self._skipped_keys.add(key)
                    warnings.warn(
                        f"MLflowCallback: '{key}' is a {type(value).__name__}, not a number; not logged."
                    )
                continue
            self._pending.append(Metric(_mlflow_key(key)[:_MLFLOW_MAX_KEY_LEN], number, timestamp, int(step)))
        if (len(self._pending) >= _MLFLOW_BATCH_SIZE
                or time.monotonic() - self._last_flush >= _MLFLOW_FLUSH_SECONDS):
            self._flush()

    def _flush(self) -> None:
        """Send buffered metrics in batches of ``_MLFLOW_BATCH_SIZE``."""
        pending, self._pending = self._pending, []
        self._last_flush = time.monotonic()
        for i in range(0, len(pending), _MLFLOW_BATCH_SIZE):
            batch = pending[i:i + _MLFLOW_BATCH_SIZE]
            if self.async_logging:
                self._client.log_batch(self.active_run_id, metrics=batch, synchronous=False)
            else:
                self._client.log_batch(self.active_run_id, metrics=batch)

    def _save_model_artifact(self, step: int) -> None:
        """Upload ``save_dir`` as the run's ``model`` artifact.

        The MLflow version of ``wandb_support.save_model_artifact``: copies
        the save directory under ``model/`` (minus ``renders/``, whose
        videos `log_render` already uploads once each), adds
        ``mlflow_config.json`` at the root when present, and tags the run
        with the step of this best checkpoint.
        """
        save_dir = Path(self.save_dir)
        for path in save_dir.rglob("*"):
            rel = path.relative_to(save_dir)
            if not path.is_file() or rel.parts[0] == "renders":
                continue  # videos are logged once each by log_render
            parent = Path("model", *rel.parts[:-1]).as_posix()
            self._client.log_artifact(self.active_run_id, str(path), artifact_path=parent)
        config_file = Path(self.save_dir) / "mlflow_config.json"
        if config_file.exists():
            self._client.log_artifact(self.active_run_id, str(config_file))
        self._client.set_tag(self.active_run_id, "best_checkpoint_step", str(step))

    def _close_at_exit(self) -> None:
        """Flush and mark an unfinished run ``KILLED`` when Python exits."""
        if self.initialized and not self._finished:
            try:
                self._finish_run(status="KILLED")
            except Exception:
                pass  # the server may already be gone; never block exit

    def _finish_run(self, status: str = "FINISHED") -> None:
        """Flush async metrics and mark the run finished, at most once per run.

        Like `WandbCallback._finish_run`, guards against the trainer firing
        the end hook once per env that finished on the final step.
        """
        if self._finished or not self.initialized:
            return
        self._flush()
        mlflow = _import_mlflow()
        flush = getattr(mlflow, "flush_async_logging", None)
        if self.async_logging and callable(flush):
            flush()
        self._client.set_terminated(self.active_run_id, status=status)
        self._finished = True
        atexit.unregister(self._close_at_exit)

    def log_render(self, video_path: str | Path, context: str, step: int, metrics: dict | None = None) -> None:
        """Upload a rendered episode video and log its metrics.

        The MLflow version of the ``wandb.Video`` logging in `Renderer`: the
        file is stored under ``renders/<context>/`` in the run's artifacts
        (MLflow's UI plays ``.mp4`` files there) and ``metrics`` are logged at
        ``step``. Does nothing if no run is open.

        Args:
            video_path: Path of the ``.mp4`` written by the renderer.
            context: ``"train"`` or ``"test"``.
            step: Trainer global step for the metrics.
            metrics: Scalars such as ``render_episode_reward``.
        """
        if not self.initialized or self._finished:
            return
        self._client.log_artifact(self.active_run_id, str(video_path), artifact_path=f"renders/{context}")
        if metrics:
            self._log(metrics, step=step)

    @property
    def run_url(self) -> str | None:
        """Web UI link to this run, or ``None`` before a run or without a server."""
        uri = self.resolved_tracking_uri or ""
        if not (uri.startswith(("http://", "https://")) and self.experiment_id and self.active_run_id):
            return None
        return f"{uri.rstrip('/')}/#/experiments/{self.experiment_id}/runs/{self.active_run_id}"

    @property
    def experiment_url(self) -> str | None:
        """Web UI link to the experiment (``project_name``), or ``None`` without a server."""
        uri = self.resolved_tracking_uri or ""
        if not (uri.startswith(("http://", "https://")) and self.experiment_id):
            return None
        return f"{uri.rstrip('/')}/#/experiments/{self.experiment_id}"

    # ------------------------------------------------------------------ hooks
    def on_train_begin(self, logs: dict, run_number: Optional[int] = None, models: Optional[list[T.nn.Module]] = None):
        """Check the tracking server and start a train run if none is open.

        Args:
            logs: Trainer config tree logged as params and ``config.json``.
            run_number: Optional run counter reused when resuming a named run.
            models: Modules whose architecture is logged, or ``None``.
        """
        self._ensure_mlflow_tracking()
        if not self.initialized:
            self.initialize_run(logs, models, run_number, run_name_prefix="train", tags=["train"], job_type="train")

    def on_train_end(self, logs=None):
        """Finish the active MLflow run when training exits. Idempotent.

        Args:
            logs (dict | None): Unused; accepted for base-class compatibility.
        """
        self._finish_run()

    def on_train_epoch_begin(self, epoch: int, logs=None):
        """No-op; MLflow logging happens on epoch end."""
        pass

    def _should_upload_artifact(self, epoch: int) -> bool:
        """Return whether the ``artifact_every`` cooldown has elapsed.

        Same rule as `WandbCallback._should_upload_artifact`.
        """
        if not self.artifact_every:
            return True
        if self._last_artifact_at is None:
            return True
        return epoch >= self._last_artifact_at + self.artifact_every

    def on_train_epoch_end(self, epoch: int, logs=None):
        """Log episode metrics and upload the model artifact if a checkpoint was saved.

        Uploads follow ``logs['saved']`` (the trainer wrote a checkpoint this
        step), throttled by ``artifact_every``, exactly like `WandbCallback`.

        Args:
            epoch: Trainer global step, used as the metric step and the
                cooldown clock.
            logs (dict | None): Episode metrics.
        """
        if logs is None:
            logs = {}
        self._log(logs, step=epoch)
        if logs.get("saved", False) and self._should_upload_artifact(epoch):
            os.makedirs(self.save_dir, exist_ok=True)
            self._save_model_artifact(epoch)
            self._last_artifact_at = epoch

    def on_train_step_begin(self, step: int, logs=None):
        """No-op; MLflow logging happens on step end."""
        pass

    def on_train_step_end(self, step: int, logs=None):
        """Log per-step training metrics.

        Args:
            step: Trainer global step used as the metric step.
            logs (dict | None): Per-step scalars.
        """
        self._log(logs or {}, step=step)

    def on_test_begin(self, logs: dict, run_number: Optional[str] = None):
        """Start a test run when none is open yet (e.g. not after training).

        The run name inherited from the agent's saved training config is
        turned into a test name (``train-3`` -> ``test-3``, see
        `_test_run_name`), like `WandbCallback.on_test_begin`.

        Args:
            logs: Trainer config tree.
            run_number: Optional run counter / suffix for the test run name.
        """
        if not self.initialized:
            self._ensure_mlflow_tracking()
            self.run_name = _test_run_name(self.run_name)
            self.initialize_run(logs=logs, run_number=run_number, run_name_prefix="test", tags=["test"], job_type="test")

    def on_test_end(self, logs=None):
        """Finish the active MLflow run when evaluation exits. Idempotent."""
        self._finish_run()

    def on_test_epoch_begin(self, epoch: int, logs=None):
        """No-op; MLflow logging happens on epoch end."""
        pass

    def on_test_epoch_end(self, epoch: int, logs=None):
        """Log episode evaluation metrics.

        Args:
            epoch: Trainer global step used as the metric step.
            logs (dict | None): Episode metrics.
        """
        self._log(logs or {}, step=epoch)

    def on_test_step_begin(self, step: int, logs=None):
        """No-op; MLflow logging happens on step end."""
        pass

    def on_test_step_end(self, step: int, logs=None):
        """Log per-step evaluation metrics.

        Args:
            step: Trainer global step used as the metric step.
            logs (dict | None): Per-step scalars.
        """
        self._log(logs or {}, step=step)

    # ------------------------------------------------------------ persistence
    def get_config(self):
        """Return a serializable ``{'type', 'config'}`` dict for this callback.

        Returns:
            config (dict): ``type`` ``"MLflowCallback"`` and the constructor
                kwargs, enough to rebuild an equivalent callback via `load`.
        """
        return {
            'type': "MLflowCallback",
            'config': {
                'project_name': self.project_name,
                'run_name': self.run_name,
                'group': self.group,
                'tags': self.tags,
                'run_id': self.run_id,
                'resume': self.resume,
                'sweep_params': self.sweep_params,
                'artifact_every': self.artifact_every,
                'extra_config': self.extra_config,
                'tracking_uri': self.tracking_uri,
                'async_logging': self.async_logging,
                'auto_start_server': self.auto_start_server,
            }
        }

    def save(self, folder: str | Path | None = None):
        """Write ``mlflow_config.json`` under ``folder`` or ``save_dir``.

        Adds a ``run`` section (tracking URI, experiment and run ids, UI
        links) next to ``type``/``config`` so other tools can find the run;
        `load` ignores it.

        Args:
            folder: Destination directory, or ``None`` for ``self.save_dir``.
        """
        path = Path(folder if folder is not None else self.save_dir) / "mlflow_config.json"
        os.makedirs(path.parent, exist_ok=True)
        data = self.get_config()
        data['run'] = {
            'tracking_uri': self.resolved_tracking_uri,
            'experiment_id': self.experiment_id,
            'run_id': self.active_run_id,
            'run_url': self.run_url,
            'experiment_url': self.experiment_url,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, config):
        """Build an `MLflowCallback` from a wrapped ``get_config()`` dict.

        Args:
            config (dict): Mapping with a ``'config'`` sub-dict of constructor kwargs.

        Returns:
            callback (MLflowCallback): New instance from ``config['config']``.
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
