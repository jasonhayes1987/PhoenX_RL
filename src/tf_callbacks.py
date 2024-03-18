import json
import requests

from tensorflow.keras.callbacks import Callback
import wandb

import wandb_support



class WandbCallback(Callback):
    """Wandb callback for Keras integration."""

    def __init__(
        self,
        project_name: str,
        run_name: str = None,
        _sweep: bool = False,
    ):
        super().__init__()
        self.project_name = project_name
        self.save_dir = None
        self.run_name = run_name
        self.model_type = None
        self._sweep = _sweep

    def on_train_begin(self, logs=None):
        """Initializes W&B run for training."""
        ##DEBUG
        # print(f"WANDB on_train_begin called: {self.model_type}")

        if not self._sweep:
            next_run_number = wandb_support.get_next_run_number(self.project_name)
            # # add num layers to config to log to wandb
            # # check which model type in order call correct classes to count number of layers
            # if self.model_type == "Reinforce" or self.model_type == "ActorCritic":
            #     logs["Policy Number Layers"] = len(self.model.policy_model.layers)
            #     logs["Critic Number Layers"] = len(self.model.value_model.layers)
            # elif self.model_type == "DDPG":
            #     logs["Actor Number Layers"] = len(self.model.actor_model.layers)
            #     logs["Critic Number Layers"] = len(self.model.critic_model.layers)

            wandb.init(
                project=self.project_name,
                name=f"train-{next_run_number}",
                tags=["train", self.model_type],
                group=f"group-{next_run_number}",
                job_type="train",
                config=logs,
            )

        # save run
        self.run_name = wandb.run.name

    def on_train_end(self, logs=None):
        """Finishes W&B run for training."""

        if not self._sweep:
            wandb.finish()

    def on_epoch_begin(self, epoch, logs=None):
        """Initializes W&B run for epoch."""

    def on_epoch_end(self, epoch, logs=None):
        """Finishes W&B run for epoch."""

        wandb.log(logs, step=epoch)
        # if best model so far (avg/100 reward), save model artifact
        if logs["best"]:
            # save model artifact
            wandb_support.save_model_artifact(self.save_dir, self.project_name, model_is_best=True)

    def on_train_batch_begin(self, batch, logs=None):
        """Initializes W&B run for training batch."""

    def on_train_batch_end(self, batch, logs=None):
        """Finishes W&B run for training batch."""

        wandb.log(logs, step=batch)

    def on_test_begin(self, logs=None):
        """Initializes W&B run for testing."""

        run_number = wandb_support.get_run_number_from_name(self.run_name)
        wandb.init(
            project=self.project_name,
            job_type="test",
            tags=["test", self.model_type],
            name=f"test-{run_number}",
            group=f"group-{run_number}",
            config=logs,
        )
        wandb.config.update({"model_type": self.model_type})  # allow_val_change=True

    def on_test_end(self, logs=None):
        """Finishes W&B run for testing."""

        wandb.finish()

    def on_test_batch_begin(self, batch, logs=None):
        """Initializes W&B run for testing batch."""

    def on_test_batch_end(self, batch, logs=None):
        """Finishes W&B run for testing batch."""

        wandb.log(logs, step=batch)

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
        # add code to save config of agent and models to wandb
        # config = {
        #     "env": agent.env.spec.id,
        #     "discount": agent.discount,
        #     # "learning_rate": agent.learning_rate,
        #     "model_type": self.model_type.replace("_", " "),
        # }

        # if self.model_type == "ActorCritic":
        #     config["policy_trace_decay"] = agent.policy_trace_decay
        #     config["value_trace_decay"] = agent.value_trace_decay

        # config.update(self._get_model_config(agent.actor_model))
        # config.update(self._get_model_config(agent.critic_model))


        return agent.get_config()

    def _get_model_config(self, model):
        """configures callback internal state for wandb integration.

        Args:
            model (rl_agents.models): The model to configure.

        Returns:
            dict: The configuration dictionary.
        """

        config = {}

        for e, layer in enumerate(model.hidden_layers):
            config[
                f"{model.__class__.__name__.split('_')[0].lower()}_units_layer_{e+1}"
            ] = layer[0]
        config[f"{model.__class__.__name__.split('_')[0].lower()}_activation"] = layer[
            1
        ]
        config[
            f"{model.__class__.__name__.split('_')[0].lower()}_optimizer"
        ] = model.optimizer.__class__.__name__.lower()

        return config
    
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

        # with open(config, "r", encoding="utf-8") as f:
        #     wandb_config = json.load(f)

        # callback = WandbCallback(
        #     project_name=wandb_config["project_name"],
        # )
        # callback.run_name = wandb_config["run_name"]

        # return callback

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

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self._episode_num += 1
        logs['episode'] = self._episode_num

        try:
            # write logs to json file to be loaded into Dash app for updating status
            with open("assets/training_data.json", 'w') as f:
                json.dump(logs, f)
        except Exception as e:
            print(f"Failed to send update to Dash app: {e}")

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_begin(self, logs=None):
        self._episode_num = 0

    def on_test_end(self, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        self._episode_num += 1
        logs['episode'] = self._episode_num

        try:
            # write logs to json file to be loaded into Dash app for updating status
            with open("assets/testing_data.json", 'w') as f:
                json.dump(logs, f)
        except Exception as e:
            print(f"Failed to send update to Dash app: {e}")

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
    agent_class = types.get(class_name)

    if agent_class:
        return agent_class.load(config)

    raise ValueError(f"Unknown agent type: {class_name}")