"""Train / test lifecycle hooks and a Weights & Biases integration callback.

`Callback` is the no-op base class; subclasses override the hooks they care
about. `Trainer` drives a subset of them at run start, after each env step,
when an episode finishes (the ``*_epoch_end`` hooks — the ``epoch`` argument
is the trainer's global step counter, not an episode index), and at run end.
`WandbCallback` logs metrics and checkpoints to W&B. Register third-party
subclasses with `register_callback`, look them up with `callback_load`, or
rebuild an instance from a ``get_config()`` dict via `load`.
"""
from pathlib import Path
import os
import json
from typing import Optional
import torch as T
import wandb

from . import wandb_support

_CALLBACK_REGISTRY: dict[str, type] = {}

# Bounds the interactive prompt inside wandb.login()'s credential fallback so a
# training run can never block indefinitely waiting on stdin.
_WANDB_LOGIN_TIMEOUT = 30


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
    an episode log carries ``best=True``.
    """

    def __init__(self, project_name: str, run_name: str = None):
        """Store the W&B project and optional run name.

        Args:
            project_name: Name of the W&B project.
            run_name: Optional run name; when omitted, ``initialize_run``
                assigns one from the project run counter.
        """
        self.project_name = project_name
        self.run_name = run_name
        self.save_dir = None
        self.initialized = False

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

    def initialize_run(self, logs: dict, models: list[T.nn.Module] = None, run_number: Optional[int] = None, run_name_prefix:Optional[str] = None, tags: list[str]=[], job_type: str="train"):
        """Start a W&B run from the trainer config and optionally watch models.

        Reads ``logs['save_dir']`` for artifact paths, allocates a run number
        when none is supplied, and calls ``wandb.init`` with the full config
        as ``config``. When ``models`` is non-empty, registers each with
        ``wandb.watch``.

        Args:
            logs: Trainer config tree; must include ``save_dir`` and
                ``agent['type']``.
            models: Modules to watch, or ``None`` / empty to skip watching.
            run_number: Explicit W&B run counter; when ``None``, taken from
                ``wandb_support.get_next_run_number``.
            run_name_prefix: Prefix for the run name
                (``"{prefix}-{run_number}"``).
            tags: Mutable tag list; the agent type is appended in place.
            job_type: W&B job type string (``"train"`` or ``"test"``).
        """
        # Set save dir
        self.save_dir = logs['save_dir']
        
        # Only get a new run number if we're initializing and none was provided
        if run_number is None:
            run_number = wandb_support.get_next_run_number(self.project_name)
        
        run = wandb.init(
            project=self.project_name,
            name=f"{run_name_prefix}-{run_number}",
            tags=tags.append(logs['agent']['type']),
            group=f"group-{run_number}",
            job_type=job_type,
            config=logs,
        )
        self.run_name = run.name
        self.initialized = True
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
            

    def on_train_end(self, logs=None):
        """Finish the active W&B run when training exits.

        Args:
            logs (dict | None): Final episode metrics (unused); accepted for
                base-class compatibility.
        """
        wandb.finish()

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

        Args:
            logs (dict | None): Final episode metrics (unused); accepted for
                base-class compatibility.
        """
        # if not self._sweep:
        wandb.finish()

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
                ``config`` sub-dict of constructor kwargs
                (``project_name``, ``run_name``).
        """
        return {
            'type': "WandbCallback",
            'config': {
                'project_name': self.project_name,
                'run_name': self.run_name,
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
