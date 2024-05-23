import os
import json
import requests


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

    def __init__(
        self,
        project_name: str,
        run_name: str = None,
        chkpt_freq: int = 100, #epochs
        _sweep: bool = False,
    ):
        super().__init__()
        self.project_name = project_name
        self.save_dir = None
        self.run_name = run_name
        self.model_type = None
        self._sweep = _sweep
        self.chkpt_freq = chkpt_freq
        # self._ckpt_counter = 1

    def on_train_begin(self, models, logs=None, run_number=None):
        """Initializes W&B run for training."""

        # self._sweep = 'WANDB_SWEEP_ID' in os.environ
        
        # ##DEBUG
        # print(f'self._sweep = {self._sweep}')


        # if not self._sweep:
        if run_number is None:
            run_number = wandb_support.get_next_run_number(self.project_name)

        wandb.init(
            project=self.project_name,
            name=f"train-{run_number}",
            tags=["train", self.model_type],
            group=f"group-{run_number}",
            job_type="train",
            config=logs,
        )
        # tell wandb to watch models to store gradients and params
        wandb.watch(models, log='all', log_freq=100, idx=1, log_graph=True)

        # save run name
        self.run_name = wandb.run.name

    def on_train_end(self, logs=None):
        """Finishes W&B run for training."""

        if not self._sweep:
            wandb.finish()

    def on_train_epoch_begin(self, epoch, logs=None):
        """Initializes W&B run for epoch."""

    def on_train_epoch_end(self, epoch, logs=None):
        """Finishes W&B run for epoch."""

        wandb.log(logs, step=epoch)
        # if best model so far (avg/100 reward), save model artifact
        if (logs["best"]) & (logs["episode"]%self.chkpt_freq == 0):
            # save model artifact
            wandb_support.save_model_artifact(self.save_dir, self.project_name, model_is_best=True)

    def on_train_step_begin(self, step, logs=None):
        """Initializes W&B run for training batch."""

    def on_train_step_end(self, step, logs=None):
        """Finishes W&B run for training batch."""

        wandb.log(logs, step=step)

    def on_test_begin(self, logs=None, run_number=None):
        """Initializes W&B run for testing."""

        if run_number is None:
            run_number = wandb_support.get_run_number_from_name(self.run_name)
        
        wandb.init(
            project=self.project_name,
            job_type="test",
            tags=["test", self.model_type],
            name=f"test-{run_number}",
            group=f"group-{run_number}",
            config=logs,
        )
        wandb.config.update({"model_type": self.model_type})

    def on_test_end(self, logs=None):
        """Finishes W&B run for testing."""

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
        """configures callback internal state for wandb integration.

        Args:
            agent (rl_agents.Agent): The agent to configure.

        Returns:
            dict: The configuration dictionary.
        """

        # set agent type
        self.model_type = type(agent).__name__
        # set save dir
        self.save_dir = agent.save_dir
        
        return agent.get_config()

    # def _get_model_config(self, model):
    #     """configures callback internal state for wandb integration.

    #     Args:
    #         model (rl_agents.models): The model to configure.

    #     Returns:
    #         dict: The configuration dictionary.
    #     """

    #     config = {}

    #     for e, layer in enumerate(model.hidden_layers):
    #         config[
    #             f"{model.__class__.__name__.split('_')[0].lower()}_units_layer_{e+1}"
    #         ] = layer[0]
    #     config[f"{model.__class__.__name__.split('_')[0].lower()}_activation"] = layer[
    #         1
    #     ]
    #     config[
    #         f"{model.__class__.__name__.split('_')[0].lower()}_optimizer"
    #     ] = model.optimizer.__class__.__name__.lower()

    #     return config
    
    def get_config(self):
        """Returns callback config for serialization"""
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'project_name': self.project_name,
                'run_name': self.run_name,
            }
        }

    def save(self, folder: str = "wandb_config.json"):
        """Save model.

        Args:
            folder (str): The folder to save the config to.
        """
        wandb_config = {
            "project_name": self.project_name,
            "run_name": self.run_name,
        }

        with open(folder, "w", encoding="utf-8") as f:
            json.dump(wandb_config, f)

    @classmethod
    def load(cls, config):
        """Load callback.

        Args:
            folder (str): The folder to load the config from.

        Returns:
            WandbCallback: The callback object.
        """

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
        self._episode_num += 1
        logs['episode'] = self._episode_num

        try:
            # write logs to json file to be loaded into Dash app for updating status
            os.makedirs("assets", exist_ok=True)
            with open("assets/training_data.json", 'w') as f:
                json.dump(logs, f)
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