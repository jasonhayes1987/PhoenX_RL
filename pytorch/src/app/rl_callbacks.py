import os
import json
import requests
import numpy as np
import torch as T


import wandb

import wandb_support


class Callback():
    """Base class for all callbacks."""
    
    def __init__(self):
        pass
    
    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_train_epoch_begin(self, epoch, logs=None):
        pass

    def on_train_epoch_end(self, epoch, logs=None):
        pass

    def on_train_step_begin(self, logs=None):
        pass

    def on_train_step_end(self, step, logs=None):
        pass

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_test_epoch_begin(self, epoch, logs=None):
        pass

    def on_test_epoch_end(self, epoch, logs=None):
        pass

    def on_test_step_begin(self, logs=None):
        pass

    def on_test_step_end(self, step, logs=None):
        pass



class WandbCallback(Callback):
    """Wandb callback for Keras integration."""

    def __init__(self, project_name:str, run_name:str=None, chkpt_freq:int=100, _sweep:bool=False):
        super().__init__()
        self.project_name = project_name
        self.run_name = run_name
        self.save_dir = None
        self.model_type = None
        self.chkpt_freq = chkpt_freq
        self._sweep = _sweep

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
        """Finishes W&B run for training."""
        # if not self._sweep:
        wandb.finish()

    def on_train_epoch_begin(self, epoch, logs=None):
        """Initializes W&B run for epoch."""

    def on_train_epoch_end(self, epoch, logs=None):
        """Finishes W&B run for epoch."""
        wandb.log(logs, step=epoch)
        if (logs["best"]) & (logs["episode"] % self.chkpt_freq == 0):
            wandb_support.save_model_artifact(self.save_dir, self.project_name, model_is_best=True)

    def on_train_step_begin(self, step, logs=None):
        """Initializes W&B run for training batch."""

    def on_train_step_end(self, step, logs=None):
        """Finishes W&B run for training batch."""
        wandb.log(logs, step=step)

    def on_test_begin(self, logs=None, run_number=None):
        if run_number == None:
            run_number = wandb_support.get_run_number_from_name(self.run_name)

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

    def on_test_epoch_begin(self, epoch, logs=None):
        """Initializes W&B run for epoch."""

    def on_test_epoch_end(self, epoch, logs=None):
        """Finishes W&B run for epoch."""
        wandb.log(logs, step=epoch)

    def on_test_step_begin(self, step, logs=None):
        """Initializes W&B run for testing batch."""

    def on_test_step_end(self, step, logs=None):
        """Finishes W&B run for testing batch."""
        wandb.log(logs, step=step)

    def _config(self, agent):
        """Configures callback internal state for wandb integration."""
        self.model_type = type(agent).__name__
        self.save_dir = agent.save_dir
        return agent.get_config()

    def get_config(self):
        """Returns callback config for serialization"""
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
        """Load callback."""
        return cls(**config)

    
class DashCallback(Callback):
    """Callback for Keras integration."""

    def __init__(self, dash_app_url):
        super().__init__()
        self.dash_app_url = dash_app_url

        # for internally counting num episodes
        self._episode_num = 0

    def on_train_begin(self, logs=None):
        self._episode_num = 0

    def on_train_end(self, logs=None):
        pass

    def on_train_epoch_begin(self, epoch, logs=None):
        pass

    def on_train_epoch_end(self, epoch, logs=None):
        # Check for 'kl_divergence' key and increment episode number if not found
        if 'kl_divergence' in logs:
            logs['episode'] = epoch
        else:
            self._episode_num += 1
            logs['episode'] = self._episode_num

        print(f'attempting to write train data file...')
        
        try:
            # Convert any tensor, float32, or other non-serializable types in the logs to serializable formats
            logs = self.convert_values_to_serializable(logs)

            # Write logs to JSON file
            os.makedirs("assets", exist_ok=True)
            with open("assets/training_data.json", 'w') as f:
                json.dump(logs, f)
            print(f'training data file saved!')
            
        except Exception as e:
            print(f"Failed to send update to Dash app: {e}")

    def on_train_step_begin(self, step, logs=None):
        pass

    def on_train_step_end(self, step, logs=None):
        pass

    def on_test_begin(self, logs=None):
        self._episode_num = 0

    def on_test_end(self, logs=None):
        pass

    def on_test_epoch_begin(self, epoch, logs=None):
        pass

    def on_test_epoch_end(self, epoch, logs=None):
        # Convert any tensor, float32, or other non-serializable types in the logs to serializable formats
        logs = self.convert_values_to_serializable(logs)
        self._episode_num += 1
        logs['episode'] = self._episode_num

        try:
            # write logs to json file to be loaded into Dash app for updating status
            with open("assets/testing_data.json", 'w') as f:
                json.dump(logs, f)
        except Exception as e:
            print(f"Failed to send update to Dash app: {e}")

    def on_test_step_begin(self, step, logs=None):
        pass

    def on_test_step_end(self, step, logs=None):
        pass

    def _config(self, agent):
        pass

    def _get_model_config(self, model):
        pass

    def get_config(self):
        """Returns callback config for serialization"""
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'dash_app_url': self.dash_app_url
            }
        }
    
    def convert_values_to_serializable(self, d):
        """Converts tensors and other non-serializable types to floats"""
        for key, value in d.items():
            if isinstance(value, T.Tensor):
                d[key] = value.item() if value.numel() == 1 else value.tolist()  # Convert scalar tensors or tensors with more elements
            elif isinstance(value, np.float32) or isinstance(value, np.float64):
                d[key] = float(value)  # Convert np.float32/float64 to standard Python float
            elif isinstance(value, (np.int32, np.int64)):
                d[key] = int(value)  # Convert numpy int types to Python int
            elif isinstance(value, dict):
                # Recursively handle nested dictionaries
                self.convert_values_to_serializable(value)
        return d

    def save(self, folder: str = "wandb_config.json"):
        pass

    @classmethod
    def load(cls, config):
        return cls(**config)
    
def load(class_name, config):
    
    types = {
        "WandbCallback": WandbCallback,
        "DashCallback": DashCallback,
    }

    # Use globals() to get a reference to the class
    callback_class = types.get(class_name)

    if callback_class:
        return callback_class.load(config)

    raise ValueError(f"Unknown agent type: {class_name}")