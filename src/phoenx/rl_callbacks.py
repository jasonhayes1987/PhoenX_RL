from pathlib import Path
import os
import json
from typing import Optional
import torch as T
import wandb

from . import wandb_support

_CALLBACK_REGISTRY: dict[str, type] = {}


# Decorator For Third Party Callbacks
def register_callback(name: str | None = None):
    """Decorator that registers a callback class under an explicit name.

    When *name* is omitted the class's own ``__name__`` is used, which matches
    the key written by every ``get_config()`` method in this module.

    Usage::

        @register_callback()              # key = "MyCallback"
        class MyCallback(Callback): ...

        @register_callback("my_alias")   # key = "my_alias"
        class MyCallback(Callback): ...
    """
    def decorator(callback_class: type) -> type:
        _CALLBACK_REGISTRY[name or callback_class.__name__] = callback_class
        return callback_class
    return decorator


def callback_load(name: str) -> type:
    """Return the callback *class* registered under *name*."""
    try:
        return _CALLBACK_REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown callback '{name}'. Available: {list(_CALLBACK_REGISTRY)}")



class Callback():
    """Base class for all callbacks in reinforcement learning.

    Methods:
        on_train_begin(logs): Called at the beginning of training.
        on_train_end(logs): Called at the end of training.
        on_train_epoch_begin(epoch, logs): Called at the beginning of each epoch during training.
        on_train_epoch_end(epoch, logs): Called at the end of each epoch during training.
        on_train_step_begin(logs): Called at the beginning of each training step.
        on_train_step_end(step, logs): Called at the end of each training step.
        on_test_begin(logs): Called at the beginning of testing.
        on_test_end(logs): Called at the end of testing.
        on_test_epoch_begin(epoch, logs): Called at the beginning of each epoch during testing.
        on_test_epoch_end(epoch, logs): Called at the end of each epoch during testing.
        on_test_step_begin(logs): Called at the beginning of each testing step.
        on_test_step_end(step, logs): Called at the end of each testing step.
    """

    def __init_subclass__(cls, register: bool = True, **kwargs):
        """Auto-register every concrete subclass by its class name."""
        super().__init_subclass__(**kwargs)
        if register:
            _CALLBACK_REGISTRY[cls.__name__] = cls
    
    def on_train_begin(self, models, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_train_epoch_begin(self, epoch: int, logs=None):
        pass

    def on_train_epoch_end(self, epoch: int, logs=None):
        pass

    def on_train_step_begin(self, logs=None):
        pass

    def on_train_step_end(self, step: int, logs=None):
        pass

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_test_epoch_begin(self, epoch: int, logs=None):
        pass

    def on_test_epoch_end(self, epoch: int, logs=None):
        pass

    def on_test_step_begin(self, logs=None):
        pass

    def on_test_step_end(self, step: int, logs=None):
        pass

    @classmethod
    def load(cls, config):

        return cls(**config)



class WandbCallback(Callback):
    """W&B integration callback for tracking and logging metrics.

    Args:
        project_name (str): Name of the W&B project.
        run_name (str, optional): Name of the specific W&B run.
        chkpt_freq (int): Frequency of saving checkpoints.
        _sweep (bool): Whether this run is part of a W&B sweep.
    """

    def __init__(self, project_name: str, run_name: str = None, _sweep: bool = False):
        # super().__init__()
        self.project_name = project_name
        self.run_name = run_name
        # self.chkpt_freq = chkpt_freq
        # self._sweep = _sweep
        self.save_dir = None
        # self.model_type = None
        self.initialized = False

    def _ensure_wandb_login(self) -> None:
        if wandb.run is not None:
            return
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            key_path = Path(__file__).with_name("wandb_api_key")
            if key_path.exists():
                api_key = key_path.read_text(encoding="utf-8").strip()
                if api_key:
                    os.environ["WANDB_API_KEY"] = api_key

        if api_key:
            wandb.login(key=api_key, relogin=False)
        else:
            raise ValueError("WANDB_API_KEY not found. Please set the WANDB_API_KEY environment variable or create a wandb_api_key file in the app directory.")

    def initialize_run(self, logs: dict, models: list[T.nn.Module] = None, run_number: Optional[int] = None, run_name_prefix:Optional[str] = None, tags: list[str]=[], job_type: str="train"):
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
        self._ensure_wandb_login()
        # if not self._sweep:
        if not self.initialized:
            self.initialize_run(logs, models, run_number, run_name_prefix="train", tags=["train"], job_type="train")
            

    def on_train_end(self, logs=None):
        wandb.finish()

    def on_train_epoch_begin(self, epoch: int, logs=None):
        pass

    def on_train_epoch_end(self, epoch: int, logs=None):
        if logs is None:
            logs = {}
        wandb.log(logs, step=epoch)
        if logs.get("best", False):
            # Create save dir if not exist
            os.makedirs(self.save_dir, exist_ok=True)
            wandb_support.save_model_artifact(self.save_dir, self.project_name, model_is_best=True)

    def on_train_step_begin(self, step: int, logs=None):
        pass

    def on_train_step_end(self, step: int, logs=None):
        if logs is None:
            logs = {}
        wandb.log(logs, step=step)

    def on_test_begin(self, logs:dict, run_number: Optional[str] = None):
        if not self.initialized:
            self.initialize_run(logs=logs, run_number=run_number, run_name_prefix="test", tags=["test"], job_type="test")

    def on_test_end(self, logs=None):
        # if not self._sweep:
        wandb.finish()

    def on_test_epoch_begin(self, epoch: int, logs=None):
        pass

    def on_test_epoch_end(self, epoch: int, logs=None):
        if logs is None:
            logs = {}
        wandb.log(logs, step=epoch)

    def on_test_step_begin(self, step: int, logs=None):
        pass

    def on_test_step_end(self, step: int, logs=None):
        if logs is None:
            logs = {}
        wandb.log(logs, step=step)

    # def _config(self, agent):
    #     """Configures callback internal state for wandb integration."""
    #     self.model_type = type(agent).__name__
    #     self.save_dir = agent.save_dir

    def get_config(self):
        return {
            'type': "WandbCallback",
            'config': {
                'project_name': self.project_name,
                'run_name': self.run_name,
                # '_sweep': self._sweep
            }
        }

    def save(self, folder: str | Path | None = None):
        """Save model."""
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

        return cls(**config['config'])

    
def load(config: dict):
    """Instantiate a callback from a config dict produced by ``get_config()``.

    Args:
        config (dict): Must contain a ``'type'`` key whose value matches a
            registered callback class name, plus a ``'config'`` sub-dict of
            constructor kwargs.

    Returns:
        Callback: An instance of the requested callback class.
    """
    cb_type = config.get("type", "")
    if cb_type not in _CALLBACK_REGISTRY:
        raise ValueError(
            f"Unknown callback type '{cb_type}'. "
            f"Available: {list(_CALLBACK_REGISTRY)}"
        )
    return _CALLBACK_REGISTRY[cb_type].load(config)