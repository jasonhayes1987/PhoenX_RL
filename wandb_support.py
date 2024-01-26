"""Adds support for W&B integration to the rl_agents package."""

# imports
import json
from pathlib import Path
import os

from tensorflow.keras.callbacks import Callback
import gymnasium as gym
import wandb

import rl_agents


def save_model_artifact(file_path: str, project_name: str, model_is_best: bool = False):
    """Save the model to W&B

    Args:
        file_path (str): The path to the model files.
        project_name (str): The name of the project.
        model_is_best (bool): Whether the model is the best model so far.
    """

    # Create a Model Version
    art = wandb.Artifact(f"{project_name}-{wandb.run.name}", type="model")
    # Add the serialized files
    art.add_file(f"{file_path}/obj_config.json", name="obj_config.json")
    art.add_dir(f"{file_path}/policy_model", name="policy_model")
    art.add_dir(f"{file_path}/value_model", name="value_model")
    # checks if there is a wandb config and if there is, log it as an artifact
    if os.path.exists(f"{file_path}/wandb_config.json"):
        art.add_file(f"{file_path}/wandb_config.json", name="wandb_config.json")
    if model_is_best:
        # If the model is the best model so far,
        #  add "best" to the aliases
        wandb.log_artifact(art, aliases=["latest", "best"])
    else:
        wandb.log_artifact(art)
    # Link the Model Version to the Collection
    wandb.run.link_artifact(art, target_path=project_name)


def load_model_from_artifact(artifact):
    """Loads the model from the specified artifact.

    Args:
        artifact (wandb.Artifact): The artifact to load the model from.

    Returns:
        rl_agents.Agent: The agent object.
    """

    # Download the artifact files to a directory
    artifact_dir = Path(artifact.download())

    return rl_agents.load_agent_from_config(artifact_dir)


def build_layers(sweep_config):
    """formats sweep_config into policy and value layers.

    Args:
        sweep_config (dict): The sweep configuration.

    Returns:
        tuple: The policy layers and value layers.
    """

    # get policy layers
    policy_layers = []
    for layer_num in range(1, sweep_config.policy_num_layers + 1):
        policy_layers.append(
            (
                sweep_config[f"policy_units_layer_{layer_num}"],
                sweep_config.policy_activation,
            )
        )
    # get value layers
    value_layers = []
    for layer_num in range(1, sweep_config.value_num_layers + 1):
        value_layers.append(
            (
                sweep_config[f"value_units_layer_{layer_num}"],
                sweep_config.value_activation,
            )
        )

    return policy_layers, value_layers


def load_model_from_run(run_name: str, project_name: str):
    """Loads the model from the specified run.

    Args:
        run_name (str): The name of the run.
        project_name (str): The name of the project.

    Returns:
        rl_agents.Agent: The agent object.
    """

    artifact = get_artifact_from_run(project_name, run_name)

    return load_model_from_artifact(artifact)


def hyperparameter_sweep(
    rl_agent,
    sweep_config,
    num_sweeps: int,
    episodes_per_sweep: int,
    save_dir: str = "models",
):
    """Runs a hyperparameter sweep of the specified agent.

    Args:
        rl_agent (rl_agents.Agent): The agent to train.
        sweep_config (dict): The sweep configuration.
        num_sweeps (int): The number of sweeps to run.
        episodes_per_sweep (int): The number of episodes to train per sweep.
        save_dir (str): The directory to save the model to.
    """

    sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config["project"])
    wandb.agent(
        sweep_id,
        function=lambda: _run_sweep(
            rl_agent, sweep_config, episodes_per_sweep, save_dir
        ),
        count=num_sweeps,
        project=sweep_config["project"],
    )
    wandb.teardown()


def _run_sweep(agent, sweep_config, episodes_per_sweep, save_dir):
    """Runs a single sweep of the hyperparameter search.

    Args:
        agent (rl_agents.Agent): The agent to train.
        sweep_config (dict): The sweep configuration.
        episodes_per_sweep (int): The number of episodes to train per sweep.
        save_dir (str): The directory to save the model to.

    Returns:
        dict: The sweep configuration.
    """

    # get next run number
    run_number = get_next_run_number(sweep_config["project"])
    wandb.init(
        project=sweep_config["project"],
        job_type="train",
        name=f"train-{run_number}",
        tags=["train", agent.__name__],
        group=f"group-{run_number}",
    )
    env = gym.make(wandb.config.env)
    policy_layers, value_layers = build_layers(wandb.config)
    rl_agent = agent.build(
        env=env,
        policy_layers=policy_layers,
        value_layers=value_layers,
        callbacks=[WandbCallback(project_name=sweep_config["project"], _sweep=True)],
        config=wandb.config,
        save_dir=save_dir,
    )

    rl_agent.train(episodes_per_sweep)


def get_run_id_from_name(project_name, run_name):
    """Returns the run ID for the specified run name.

    Args:
        project_name (str): The name of the project.
        run_name (str): The name of the run.

    Returns:
        str: The run ID.
    """

    api = wandb.Api()
    # Fetch all runs in the project
    runs = api.runs(f"{api.default_entity}/{project_name}")
    # Iterate over the runs and find the one with the matching name
    for run in runs:
        if run.name == run_name:
            return run.id

    # If we get here, no run has the given name
    return None


def get_run_number_from_name(run_name):
    """Extracts the run number from the run name.

    Args:
    run_name (str): The run name, e.g., 'train-4'.

    Returns:
    int: The extracted run number.
    """
    try:
        return int(run_name.split("-")[-1])

    except (IndexError, ValueError) as exc:
        raise ValueError(
            "Invalid run name format. Expected format 'train-X' where X is an integer."
        ) from exc


def get_next_run_number(project_name):
    """Returns the next run number for the specified project.

    Args:
        project_name (str): The name of the project.

    Returns:
        int: The next run number.
    """
    api = wandb.Api()
    # Get the list of runs from the project
    runs = api.runs(f"jasonhayes1987/{project_name}")
    if runs:
        # Extract the run numbers and find the maximum
        run_numbers = [int(run.name.split("-")[-1]) for run in runs]
        next_run_number = max(run_numbers) + 1
    else:
        next_run_number = 1

    return next_run_number


def get_run(project_name, run_name):
    """Returns the specified run.

    Args:
    project_name (str): The name of the project.
    run_name (str): The name of the run.

    Returns:
    wandb.Run: The run object.
    """

    api = wandb.Api()
    # get the runs ID
    run_id = get_run_id_from_name(project_name, run_name)

    # Fetch the run using the project and run name
    run_path = f"{api.default_entity}/{project_name}/{run_id}"
    run = api.run(run_path)

    return run


def get_artifact_from_run(
    project_name, run_name, artifact_type: str = "model", version="latest"
):
    """Returns the specified artifact from the specified run.

    Args:
    project_name (str): The name of the project.
    run_name (str): The name of the run.
    artifact_type (str): The type of artifact to fetch.
    version (str): The version of the artifact to fetch.

    Returns:
    wandb.Artifact: The artifact object.
    """
    api = wandb.Api()
    # Get the run
    run = get_run(project_name, run_name)

    # Get the list of artifacts linked to this run
    linked_artifacts = run.logged_artifacts()

    # Find the artifact of the specified type
    artifact_name = None
    for artifact in linked_artifacts:
        if artifact.type == artifact_type and version in artifact.aliases:
            artifact_name = artifact.name
            break
    if artifact_name is None:
        raise ValueError("No artifact of the specified type found in the run")

    # Construct the artifact path
    artifact_path = f"{api.default_entity}/{project_name}/{artifact_name}"

    # Fetch the artifact
    artifact = api.artifact(artifact_path, type=artifact_type)

    return artifact


def get_projects():
    """Returns the list of projects."""
    api = wandb.Api()
    projects = api.projects()

    return projects


def get_runs(project_name):
    """Returns the list of runs for the specified project.

    Args:
        project_name (str): The name of the project.

    Returns:
        list: The list of runs.
    """

    api = wandb.Api()

    runs = api.runs(f"{api.default_entity}/{project_name}")

    return runs


def delete_all_runs(project_name, delete_artifacts: bool = True):
    """Deletes all runs for the specified project.

    Args:
    project_name (str): The name of the project.
    delete_artifacts (bool): Whether to delete the artifacts associated with the runs.
    """

    api = wandb.Api()
    wandb.finish()
    runs = api.runs(f"{api.default_entity}/{project_name}")
    for run in runs:
        print(f"Deleting run: {run.name}")
        run.delete()


## NOT WORKING
# def delete_all_sweeps(project_name):
#     api = wandb.Api()
#     wandb.finish()
#     project = api.project(f"{api.default_entity}/{project_name}")
#     print(f"Deleting all sweeps in project: {project_name}")
#     sweeps = project.sweeps()
#     print(f"sweeps: {sweeps}")
#     for sweep in sweeps:
#         print(f"Deleting sweep: {sweep.id}")
#         sweep.delete()

## NOT WORKING
# def delete_artifacts(project_name, only_empty: bool = True):
#     api = wandb.Api()
#     artifacts = api.artifacts(f"{api.default_entity}/{project_name}/model", per_page=1000)

#     for artifact in artifacts:
#         # Fetch the artifact to get detailed info
#         artifact = artifact.fetch()
#         if only_empty:
#             if len(artifact.manifest.entries) == 0:
#                 print(f"Deleting empty artifact: {artifact.name}")
#                 artifact.delete()
#             else:
#                 print(f"Artifact {artifact.name} is not empty and will not be deleted.")
#         else:
#             print(f"Deleting artifact: {artifact.name}")
#             artifact.delete()


def custom_wandb_init(*args, **kwargs):
    """Initializes a W&B run and prints the run ID."""
    print("Initializing W&B run...")
    run = wandb.init(*args, **kwargs)
    print(f"Run initialized with ID: {run.id}")

    return run


def custom_wandb_finish():
    """Finishes a W&B run."""
    print("Finishing W&B run...")
    wandb.finish()
    print("Run finished.")


def flush_cache():
    """Flushes the W&B cache."""
    api = wandb.Api()
    api.flush()


class WandbCallback(Callback):
    """Wandb callback for Keras integration."""

    def __init__(
        self,
        project_name: str,
        _sweep: bool = False,
    ):
        super().__init__()
        self.project_name = project_name
        self.save_dir = None
        self.run_name = None
        self.model_type = None
        self._sweep = _sweep

    def on_train_begin(self, logs=None):
        """Initializes W&B run for training."""

        if not self._sweep:
            next_run_number = get_next_run_number(self.project_name)
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
            save_model_artifact(self.save_dir, self.project_name, model_is_best=True)

    def on_train_batch_begin(self, batch, logs=None):
        """Initializes W&B run for training batch."""

    def on_train_batch_end(self, batch, logs=None):
        """Finishes W&B run for training batch."""

        wandb.log(logs, step=batch)

    def on_test_begin(self, logs=None):
        """Initializes W&B run for testing."""

        run_number = get_run_number_from_name(self.run_name)
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
    def load(cls, folder):
        """Load model.

        Args:
            folder (str): The folder to load the config from.

        Returns:
            WandbCallback: The callback object.
        """

        with open(folder, "r", encoding="utf-8") as f:
            wandb_config = json.load(f)

        callback = WandbCallback(
            project_name=wandb_config["project_name"],
        )
        callback.run_name = wandb_config["run_name"]

        return callback
