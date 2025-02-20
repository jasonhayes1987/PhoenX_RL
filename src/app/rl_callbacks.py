import os
import json
import numpy as np
import torch as T
import wandb

import wandb_support


class Callback():
    """
    Base class for all callbacks in reinforcement learning.

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
    
    def on_train_begin(self, logs=None):
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



class WandbCallback(Callback):
    """
    W&B integration callback for tracking and logging metrics.

    Args:
        project_name (str): Name of the W&B project.
        run_name (str, optional): Name of the specific W&B run.
        chkpt_freq (int): Frequency of saving checkpoints.
        _sweep (bool): Whether this run is part of a W&B sweep.
    """

    def __init__(self, project_name: str, run_name: str = None, chkpt_freq: int = 100, _sweep: bool = False):
        super().__init__()
        self.project_name = project_name
        self.run_name = run_name
        self.chkpt_freq = chkpt_freq
        self._sweep = _sweep
        self.save_dir = None
        self.model_type = None

    def on_train_begin(self, models, logs=None):
        if not self._sweep:
            run_number = wandb_support.get_next_run_number(self.project_name)
            run = wandb.init(
                project=self.project_name,
                name=f"train-{run_number}",
                tags=["train", self.model_type],
                group=f"group-{run_number}",
                job_type="train",
                config=logs,
            )
            self.run_name = run.name
        wandb.watch(models, log='all', log_freq=100, idx=1, log_graph=True)

    def on_train_end(self, logs=None):
        wandb.finish()

    def on_train_epoch_begin(self, epoch: int, logs=None):
        pass

    def on_train_epoch_end(self, epoch: int, logs=None):
        wandb.log(logs, step=epoch)
        if (logs["best"]) & (logs["episode"] % self.chkpt_freq == 0):
            wandb_support.save_model_artifact(self.save_dir, self.project_name, model_is_best=True)

    def on_train_step_begin(self, step: int, logs=None):
        pass

    def on_train_step_end(self, step: int, logs=None):
        wandb.log(logs, step=step)

    def on_test_begin(self, logs=None, run_number: int = None):
        if run_number is None:
            try:
                run_number = wandb_support.get_run_number_from_name(self.run_name)
            except AttributeError:
                run_number = wandb_support.get_next_run_number(self.project_name)
        run = wandb.init(
            project=self.project_name,
            job_type="test",
            tags=["test", self.model_type],
            name=f"test-{run_number}",
            group=f"group-{run_number}",
            config=logs,
        )
        wandb.config.update({"model_type": self.model_type})
        self.run_name = run.name

    def on_test_end(self, logs=None):
        if not self._sweep:
            wandb.finish()

    def on_test_epoch_begin(self, epoch: int, logs=None):
        pass

    def on_test_epoch_end(self, epoch: int, logs=None):
        wandb.log(logs, step=epoch)

    def on_test_step_begin(self, step: int, logs=None):
        pass

    def on_test_step_end(self, step: int, logs=None):
        wandb.log(logs, step=step)

    def _config(self, agent):
        """Configures callback internal state for wandb integration."""
        self.model_type = type(agent).__name__
        self.save_dir = agent.save_dir
        return agent.get_config()

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'project_name': self.project_name,
                'run_name': self.run_name,
                'chkpt_freq': self.chkpt_freq,
                '_sweep': self._sweep
            }
        }

    def save(self, folder: str = "wandb_config.json"):
        """Save model."""
        wandb_config = self.get_config()
        with open(folder, "w", encoding="utf-8") as f:
            json.dump(wandb_config, f)

    @classmethod
    def load(cls, config):
        return cls(**config)

    
class DashCallback(Callback):
    """
    Callback for sending training/testing data to a Dash app.

    Args:
        dash_app_url (str): URL of the Dash app.
    """
    def __init__(self, dash_app_url: str):
        super().__init__()
        self.dash_app_url = dash_app_url
        self._episode_num = 0

    def on_train_begin(self, logs=None):
        self._episode_num = 0

    def on_train_end(self, logs=None):
        pass

    def on_train_epoch_begin(self, epoch: int, logs=None):
        pass

    def on_train_epoch_end(self, epoch: int, logs=None):
        logs = self.convert_values_to_serializable(logs)
        self._episode_num += 1
        logs['episode'] = self._episode_num

        try:
            os.makedirs("assets", exist_ok=True)
            with open("assets/training_data.json", 'w') as f:
                json.dump(logs, f)
        except Exception as e:
            print(f"Failed to save training data to Dash app: {e}")

    def on_train_step_begin(self, step: int, logs=None):
        pass

    def on_train_step_end(self, step: int, logs=None):
        pass

    def on_test_begin(self, logs=None):
        self._episode_num = 0

    def on_test_end(self, logs=None):
        pass

    def on_test_epoch_begin(self, epoch: int, logs=None):
        pass

    def on_test_epoch_end(self, epoch: int, logs=None):
        logs = self.convert_values_to_serializable(logs)
        self._episode_num += 1
        logs['episode'] = self._episode_num

        try:
            with open("assets/testing_data.json", 'w') as f:
                json.dump(logs, f)
        except Exception as e:
            print(f"Failed to save testing data to Dash app: {e}")

    def on_test_step_begin(self, step: int, logs=None):
        pass

    def on_test_step_end(self, step: int, logs=None):
        pass

    def _config(self, agent):
        pass

    # def _get_model_config(self, model):
    #     pass

    def convert_values_to_serializable(self, d: dict):
        for key, value in d.items():
            if isinstance(value, T.Tensor):
                d[key] = value.item() if value.numel() == 1 else value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                d[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                d[key] = int(value)
            elif isinstance(value, dict):
                self.convert_values_to_serializable(value)
        return d

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'dash_app_url': self.dash_app_url
            }
        }

    def save(self, folder: str = "wandb_config.json"):
        pass

    @classmethod
    def load(cls, config: dict):
        return cls(**config)
    
def load(class_name: str, config: dict):
    """
    Load a callback class from its name and configuration.

    Args:
        class_name (str): Name of the callback class.
        config (dict): Configuration dictionary for the callback.

    Returns:
        Callback: An instance of the callback class.
    """
    types = {
        "WandbCallback": WandbCallback,
        "DashCallback": DashCallback,
    }

    if class_name in types:
        return types[class_name].load(config)

    raise ValueError(f"Unknown callback type: {class_name}")