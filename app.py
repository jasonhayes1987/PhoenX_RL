"""Streamlit app for RL Model training and testing"""

# imports
import tkinter as tk
from tkinter import filedialog

import streamlit as st
from streamlit_option_menu import option_menu
import gymnasium as gym
import wandb

import wandb_support
import rl_agents
import models
import helper
import streamlit_support as stsp


def main():
    """Main function of the App."""

    st.title("RL Model Training and Testing App")
    with st.sidebar:
        app_mode = option_menu(
            menu_title="Main Menu",  # required
            options=[
                "Home",
                "Build Model",
                "Train Model",
                "Test Model",
                "Hyperparameter Search",
                "WandB Utils",
            ],  # required
            icons=[
                "house",
                "gear",
                "play-circle",
                "check2-circle",
                "search",
                "tools",
            ],  # optional, using 'search' for Hyperparameter Search and 'tools' for WandB Utils
            menu_icon="cast",  # optional
            default_index=0,  # optional
        )

    if app_mode == "Home":
        home()
    elif app_mode == "Build Model":
        build_model()
    elif app_mode == "Train Model":
        train_model()
    elif app_mode == "Test Model":
        test_model()
    elif app_mode == "Hyperparameter Search":
        setup_sweeps()
    elif app_mode == "WandB Utils":
        wandb_utils()


def load_agent():
    """Load an agent from a folder or WandB artifact."""

    st.subheader("Load Agent for Directory")
    # create a button for load from folder
    if st.button("Load from Folder"):
        load_path = select_folder()
        if load_path:
            # Logic to load agent from the folder
            agent = rl_agents.load_agent_from_config(load_path)
            st.session_state["loaded_agent"] = agent
            st.success(f"Agent loaded from {load_path}")

    st.subheader("Load from WandB Artifact")
    project = st.selectbox(
        "Select Project", [project.name for project in wandb_support.get_projects()]
    )
    run_names = [
        run.name for run in wandb_support.get_runs(project) if "test" not in run.name
    ]
    run_name = st.selectbox("Select Run", run_names)
    if st.button("Load from WandB Artifact"):
        agent = wandb_support.load_model_from_run(run_name, project)
        st.session_state["loaded_agent"] = agent
        st.success(f"Agent loaded from {project} {run_name}")


def home():
    pass


def build(agent_copy=None):
    """Build an agent based on the user's selections.

    Args:
        agent_copy (rl_agents.Agent): An agent to copy the configuration from.

    Returns:
        An instance of the selected agent class.
    """

    if agent_copy:
        if st.session_state.model_type == "Reinforce":
            policy_model = models.PolicyModel(
                env=gym.make(st.session_state.env),
                hidden_layers=agent_copy.policy_model.hidden_layers,
                optimizer=helper.get_optimizer_by_name(
                    agent_copy.policy_model.optimizer.__class__.__name__.lower()
                ),
            )
            value_model = models.ValueModel(
                env=gym.make(st.session_state.env),
                hidden_layers=agent_copy.value_model.hidden_layers,
                optimizer=helper.get_optimizer_by_name(
                    agent_copy.value_model.optimizer.__class__.__name__.lower()
                ),
            )
            agent = rl_agents.Reinforce(
                env=gym.make(st.session_state.env),
                policy_model=policy_model,
                value_model=value_model,
                learning_rate=agent_copy.learning_rate,
                discount=agent_copy.discount,
                callbacks=agent_copy.callbacks,
                save_dir=agent_copy.save_dir,
            )
        elif st.session_state.model_type == "Actor Critic":
            policy_model = models.PolicyModel(
                env=gym.make(st.session_state.env),
                hidden_layers=agent_copy.policy_model.hidden_layers,
                optimizer=helper.get_optimizer_by_name(
                    agent_copy.policy_model.optimizer.__class__.__name__.lower()
                ),
            )
            value_model = models.ValueModel(
                env=gym.make(st.session_state.env),
                hidden_layers=agent_copy.value_model.hidden_layers,
                optimizer=helper.get_optimizer_by_name(
                    agent_copy.value_model.optimizer.__class__.__name__.lower()
                ),
            )
            agent = rl_agents.ActorCritic(
                env=gym.make(st.session_state.env),
                policy_model=policy_model,
                value_model=value_model,
                learning_rate=agent_copy.learning_rate,
                discount=agent_copy.discount,
                policy_trace_decay=agent_copy.policy_trace_decay,
                value_trace_decay=agent_copy.value_trace_decay,
                callbacks=agent_copy.callbacks,
                save_dir=agent_copy.save_dir,
            )

    else:
        # set a defualt env to build
        env_name = "CartPole-v1"
        policy_optimizer = helper.get_optimizer_by_name(
            st.session_state.policy_optimizer
        )
        value_optimizer = helper.get_optimizer_by_name(st.session_state.value_optimizer)
        policy_layers = models.build_layers(
            st.session_state.policy_units_per_layer, st.session_state.policy_activation
        )
        
        value_layers = models.build_layers(
            st.session_state.value_units_per_layer, st.session_state.value_activation
        )
        
        if st.session_state.model_type == "Reinforce":
            policy_model = models.PolicyModel(
                env=gym.make(env_name),
                hidden_layers=policy_layers,
                optimizer=policy_optimizer,
            )
            value_model = models.ValueModel(
                env=gym.make(env_name),
                hidden_layers=value_layers,
                optimizer=value_optimizer,
            )
            agent = rl_agents.Reinforce(
                env=gym.make(env_name),
                policy_model=policy_model,
                value_model=value_model,
                learning_rate=st.session_state.learning_rate,
                discount=st.session_state.discount,
                callbacks=st.session_state.callback_objs,
                save_dir=st.session_state.folder_path,
            )
        elif st.session_state.model_type == "Actor Critic":
            policy_model = models.PolicyModel(
                env=gym.make(env_name),
                hidden_layers=policy_layers,
                optimizer=policy_optimizer,
            )
            value_model = models.ValueModel(
                env=gym.make(env_name),
                hidden_layers=value_layers,
                optimizer=value_optimizer,
            )
            agent = rl_agents.ActorCritic(
                env=gym.make(env_name),
                policy_model=policy_model,
                value_model=value_model,
                learning_rate=st.session_state.learning_rate,
                discount=st.session_state.discount,
                policy_trace_decay=st.session_state.policy_trace_decay,
                value_trace_decay=st.session_state.value_trace_decay,
                callbacks=st.session_state.callback_objs,
                save_dir=st.session_state.folder_path,
            )

    # agent = rl_agents.get_agent_class_from_type(st.session_state.model_type)

    return agent


def build_model():
    """Build a model based on the user's selections."""

    st.header("Build Model")
    # Model type selection
    model_type = st.selectbox(
        "Select Model Type",
        ["Reinforce", "Actor Critic"],
        help="Choose a model type. Reinforce is simpler and Actor Critic has separate policy/value models.",
    )
    # save variable to session state
    st.session_state.model_type = model_type

    # Common parameters for both models
    learning_rate = st.number_input(
        "Learning Rate",
        format="%.6f",
        step=1e-6,
        value=1e-5,
        help="The learning rate controls how much to adjust the model's weights with respect to the gradient loss at each iteration. A smaller value may lead to more precise convergence but can slow down the training process.",
    )
    discount = st.number_input(
        "Discount Factor",
        min_value=0.0,
        max_value=1.0,
        value=0.99,
        step=0.01,
        help="The discount factor determines the importance of future rewards. A value of 0 will make the agent short-sighted by only considering current rewards, while a value closer to 1 will make it strive for long-term high rewards.",
    )
    # save variables to session state
    st.session_state.learning_rate = learning_rate
    st.session_state.discount = discount

    # Policy Model Configuration
    st.subheader(
        "Policy Model Configuration",
        help="The Policy Model network estimates the probabilities of taking actions in a given state.",
    )
    policy_hidden_layers = st.number_input(
        "Number of Hidden Layers",
        min_value=1,
        max_value=10,
        value=1,
        help="This sets the depth of the neural network. More layers can capture complex features but may increase the risk of overfitting and require more data and training time.",
    )
    policy_units_per_layer = get_layer_units_input("Policy", policy_hidden_layers)
    policy_activation = st.selectbox(
        "Policy Activation Function",
        ["relu", "tanh", "sigmoid"],
        help="An activation function defines how the neurons in a neural network transform input signals into output signals, adding non-linearity to the learning process. Choose the activation function for the neurons. 'Relu' outputs the input directly if it's positive, otherwise, it outputs zero, and is generally a good default, 'tanh' centers the output between -1 and 1, and 'sigmoid' squashes the output to be between 0 and 1.",
    )
    policy_optimizer = st.selectbox(
        "Policy Optimizer",
        ["adam", "sgd", "rmsprop"],
        help="An optimizer is an algorithm that adjusts the weights of the network to minimize the loss function, guiding how the model learns from the data. Select the optimization algorithm for the policy network. Adam is a popular choice that combines the benefits of two other extensions of stochastic gradient descent.",
    )
    # save variables to session state
    st.session_state.policy_hidden_layers = policy_hidden_layers
    st.session_state.policy_units_per_layer = policy_units_per_layer
    st.session_state.policy_activation = policy_activation
    st.session_state.policy_optimizer = policy_optimizer

    # Value Model Configuration
    st.subheader(
        "Value Model Configuration",
        help=" The Value Model network estimates the value of being in a given state.",
    )
    value_hidden_layers = st.number_input(
        "Number of Hidden Layers (Value Model)",
        min_value=1,
        max_value=10,
        value=1,
        help="This sets the depth of the neural network. More layers can capture complex features but may increase the risk of overfitting and require more data and training time.",
    )
    value_units_per_layer = get_layer_units_input("Value", value_hidden_layers)
    value_activation = st.selectbox(
        "Value Activation Function",
        ["relu", "tanh", "sigmoid"],
        help="An activation function defines how the neurons in a neural network transform input signals into output signals, adding non-linearity to the learning process. Choose the activation function for the neurons. 'Relu' outputs the input directly if it's positive, otherwise, it outputs zero, and is generally a good default, 'tanh' centers the output between -1 and 1, and 'sigmoid' squashes the output to be between 0 and 1.",
    )
    value_optimizer = st.selectbox(
        "Value Optimizer",
        ["adam", "sgd", "rmsprop"],
        help="An optimizer is an algorithm that adjusts the weights of the network to minimize the loss function, guiding how the model learns from the data. Select the optimization algorithm for the value network. Adam is a popular choice that combines the benefits of two other extensions of stochastic gradient descent.",
    )
    # save variables to session state
    st.session_state.value_hidden_layers = value_hidden_layers
    st.session_state.value_units_per_layer = value_units_per_layer
    st.session_state.value_activation = value_activation
    st.session_state.value_optimizer = value_optimizer

    # Specific parameters for Actor Critic
    if model_type == "Actor Critic":
        policy_trace_decay = st.number_input(
            "Policy Trace Decay", min_value=0.0, max_value=1.0, value=0.0, step=0.01
        )
        value_trace_decay = st.number_input(
            "Value Trace Decay", min_value=0.0, max_value=1.0, value=0.0, step=0.01
        )
        # save variables to session state
        st.session_state.policy_trace_decay = policy_trace_decay
        st.session_state.value_trace_decay = value_trace_decay

    # Callbacks
    callbacks = st.multiselect(
        "Select Callbacks",
        ["Weights & Biases"],
        help="Callbacks are functions applied at certain stages of the training process, such as at the end of an epoch. Choose 'Weights & Biases' to enable experiment tracking and visualization.",
    )
    # if callbacks are selected
    if callbacks:
        # Weights & Biases Callback
        if "Weights & Biases" in callbacks:
            login_to_wandb()
            project_list = (
                get_projects()
            )  # Ensure this function gets the list of projects
            project = st.selectbox("Select Project", project_list)
            st.session_state.project = project

            # Proceed only if a project is selected
            if project:
                st.session_state.callback_objs = [
                    get_callback(name) for name in callbacks
                ]
                # callback_objs = []
                # for callback_name in callbacks:
                #     callback_obj = get_callback(callback_name)
                #     if callback_obj:
                #         callback_objs.append(callback_obj)
                # st.session_state.callback_objs = callback_objs
        else:
            st.session_state.callback_objs = [get_callback(name) for name in callbacks]
    else:
        st.session_state.callbacks = None
        st.session_state.callback_objs = None

    # Save directory using file selector
    st.subheader("Select Save Directory")
    selected_folder_path = st.session_state.get("folder_path", None)
    folder_select_button = st.button(
        "Select Folder",
        help="Choose the directory where your trained model and related files will be saved. It is recommended to choose a dedicated folder to keep your project organized.",
    )
    if folder_select_button:
        selected_folder_path = select_folder()
        st.session_state.folder_path = selected_folder_path
    if selected_folder_path:
        st.write("Selected folder path:", selected_folder_path)

    # Button to build model
    st.subheader("Build Model")
    if st.button("Build Model"):
        # build model
        agent = build()
        # save model
        agent.save()
        # display success message
        st.success(f"{st.session_state.model_type} Model built successfully!")


def get_callback(callback_name):
    """Returns a callback object based on the name provided.

    Args:
        callback_name (str): The name of the callback.

    Returns:
        An instance of the requested callback.
    """

    callbacks = {
        "Weights & Biases": wandb_support.WandbCallback(st.session_state.project),
    }

    return callbacks[callback_name]


def get_projects():
    """Returns a list of projects from the W&B API."""

    api = wandb.Api()

    return [project.name for project in api.projects()]


def get_layer_units_input(model_prefix, num_layers):
    """Returns a list of layer units based on the user's input.

    Args:
        model_prefix (str): The prefix to use for the model type.
        num_layers (int): The number of layers to configure.

    Returns:
        A list of layer units.
    """

    layer_units = []
    for i in range(1, num_layers + 1):
        units = st.number_input(
            f"{model_prefix} Model - Units in Layer {i}",
            min_value=1,
            max_value=1024,
            value=100,
            help=f"Specify the number of neurons in layer {i} of the {model_prefix} Model. More neurons can represent more complex functions but may also lead to overfitting.",
        )
        layer_units.append(units)

    return layer_units


def select_folder():
    """Returns the path of the selected folder."""

    root = tk.Tk()
    root.attributes("-topmost", True)  # Make the root window always on top
    # root.update()  # Update the window
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory(master=root)
    root.destroy()

    return folder_path


def login_to_wandb():
    """Logs in to Weights & Biases using the API key provided by the user."""

    # check if the user is already logged in
    if wandb.api.api_key:
        st.success("You are already logged in to Weights & Biases.")
    else:
        # Check if the W&B API key is stored in Streamlit secrets
        api_key = st.secrets["wandb_api_key"] if "wandb_api_key" in st.secrets else None

        # If API key is not in secrets, prompt the user to enter it
        if not api_key:
            api_key = st.text_input(
                "Enter your Weights & Biases API key",
                type="password",
                help="Input your API key from Weights & Biases to log in and sync your training runs with the cloud for tracking and analysis. Your key is kept secure and not stored after the session ends.",
            )
            # Store the API key in the session state for the duration of the session (optional)
            if api_key:
                st.session_state["wb_api_key"] = api_key

        # Proceed with the login process if API key is available
        if api_key:
            try:
                wandb.login(key=api_key)
                st.success("Logged in to Weights & Biases successfully.")
            except (ValueError, ConnectionError) as e:
                st.error(f"Failed to log in to Weights & Biases: {e}")


def train_model():
    """Train a model based on the user's selections."""

    st.header("Train Model")
    # load an agent
    agent = None
    # if not agent:
    load_agent()

    # Select gym environment
    env_data = {
        "CartPole-v0": {
            "description": "Balance a pole on a cart; the goal is to prevent it from falling.",
            "gif_url": "https://gymnasium.farama.org/_images/cart_pole.gif",
        },
        "CartPole-v1": {
            "description": "Similar to CartPole-v0 but with a different set of parameters for a harder challenge.",
            "gif_url": "https://gymnasium.farama.org/_images/cart_pole.gif",
        },
        "MountainCar-v0": {
            "description": "A car is on a one-dimensional track between two mountains; the goal is to drive up the mountain on the right.",
            "gif_url": "https://gymnasium.farama.org/_images/mountain_car.gif",
        },
        "MountainCarContinuous-v0": {
            "description": "A continuous control version of the MountainCar environment.",
            "gif_url": "https://gymnasium.farama.org/_images/mountain_car.gif",
        },
        "Pendulum-v1": {
            "description": "Control a frictionless pendulum to keep it upright.",
            "gif_url": "https://gymnasium.farama.org/_images/pendulum.gif",
        },
        "Acrobot-v1": {
            "description": "A 2-link robot that swings up to reach a given height.",
            "gif_url": "https://gymnasium.farama.org/_images/acrobot.gif",
        },
        "LunarLander-v2": {
            "description": "Lunar Lander description",
            "gif_url": "https://gymnasium.farama.org/_images/lunar_lander.gif",
        },
        "LunarLanderContinuous-v2": {
            "description": "A continuous control version of the LunarLander environment.",
            "gif_url": "https://gymnasium.farama.org/_images/lunar_lander.gif",
        },
        "BipedalWalker-v3": {
            "description": "Control a two-legged robot to walk through rough terrain without falling.",
            "gif_url": "https://gymnasium.farama.org/_images/bipedal_walker.gif",
        },
        "BipedalWalkerHardcore-v3": {
            "description": "A more challenging version of BipedalWalker with harder terrain and obstacles.",
            "gif_url": "https://gymnasium.farama.org/_images/bipedal_walker.gif",
        },
        "CarRacing-v2": {
            "description": "A car racing environment where the goal is to complete a track as quickly as possible.",
            "gif_url": "https://gymnasium.farama.org/_images/car_racing.gif",
        },
        "Blackjack-v1": {
            "description": "A classic Blackjack card game environment.",
            "gif_url": "https://gymnasium.farama.org/_images/blackjack1.gif",
        },
        "FrozenLake-v1": {
            "description": "Navigate a grid world to reach a goal without falling into holes, akin to crossing a frozen lake.",
            "gif_url": "https://gymnasium.farama.org/_images/frozen_lake.gif",
        },
        "FrozenLake8x8-v1": {
            "description": "An 8x8 version of the FrozenLake environment, providing a larger and more complex grid.",
            "gif_url": "https://gymnasium.farama.org/_images/frozen_lake.gif",
        },
        "CliffWalking-v0": {
            "description": "A grid-based environment where the agent must navigate cliffs to reach a goal.",
            "gif_url": "https://gymnasium.farama.org/_images/cliff_walking.gif",
        },
        "Taxi-v3": {
            "description": "A taxi must pick up and drop off passengers at designated locations.",
            "gif_url": "https://gymnasium.farama.org/_images/taxi.gif",
        },
        "Reacher-v4": {
            "description": "Control a robotic arm to reach a target location",
            "gif_url": "https://gymnasium.farama.org/_images/reacher.gif",
        },
        "Pusher-v4": {
            "description": "A robot arm needs to push objects to a target location.",
            "gif_url": "https://gymnasium.farama.org/_images/pusher.gif",
        },
        "InvertedPendulum-v4": {
            "description": "Balance a pendulum in the upright position on a moving cart",
            "gif_url": "https://gymnasium.farama.org/_images/inverted_pendulum.gif",
        },
        "InvertedDoublePendulum-v4": {
            "description": "A more complex version of the InvertedPendulum with two pendulums to balance.",
            "gif_url": "https://gymnasium.farama.org/_images/inverted_double_pendulum.gif",
        },
        "HalfCheetah-v4": {
            "description": "Control a 2D cheetah robot to make it run as fast as possible.",
            "gif_url": "https://gymnasium.farama.org/_images/half_cheetah.gif",
        },
        "Hopper-v4": {
            "description": "Make a two-dimensional one-legged robot hop forward as fast as possible.",
            "gif_url": "https://gymnasium.farama.org/_images/hopper.gif",
        },
        "Swimmer-v4": {
            "description": "Control a snake-like robot to make it swim through water.",
            "gif_url": "https://gymnasium.farama.org/_images/swimmer.gif",
        },
        "Walker2d-v4": {
            "description": "A bipedal robot walking simulation aiming to move forward as fast as possible.",
            "gif_url": "https://gymnasium.farama.org/_images/walker2d.gif",
        },
        "Ant-v4": {
            "description": "Control a four-legged robot to explore a terrain.",
            "gif_url": "https://gymnasium.farama.org/_images/ant.gif",
        },
        "Humanoid-v4": {
            "description": "A two-legged humanoid robot that learns to walk and balance.",
            "gif_url": "https://gymnasium.farama.org/_images/humanoid.gif",
        },
        "HumanoidStandup-v4": {
            "description": "The goal is to make a humanoid stand up from a prone position.",
            "gif_url": "https://gymnasium.farama.org/_images/humanoid_standup.gif",
        },
    }
    all_envs = (
        get_all_gym_envs()
    )  # This function should query gym's registry for environments
    env_name = st.selectbox("Select Gym Environment", all_envs)
    # Display the description and GIF for the selected environment
    if env_name in env_data:
        st.write(env_data[env_name]["description"])
        st.image(env_data[env_name]["gif_url"], width=300)  # Adjust width as needed
        st.session_state.env = env_name
        st.session_state.loaded_agent = build(agent_copy=st.session_state.loaded_agent)

    # set number of episodes
    num_episodes = st.number_input("Number of Episodes", min_value=1, value=1000)
    # Render options
    render_option = st.checkbox("Render Training Episodes")
    render_freq = None
    if render_option:
        render_freq = st.number_input(
            "Render Frequency (every 'n' episodes)", min_value=1, value=10
        )

    if st.button("Start Training"):
        # add progress bar and terminal output placeholders
        progress_bar = st.progress(0)
        terminal_output = st.empty()
        # add streamlit callback to agent callback list
        streamlit_callback = stsp.StreamlitCallback(
            progress_bar, terminal_output, num_episodes
        )
        st.session_state.loaded_agent.callback_list.append(streamlit_callback)
        # start training
        st.session_state.loaded_agent.train(
            num_episodes, render_option, render_freq=render_freq
        )


def get_all_gym_envs():
    """Returns a list of all gym environments."""

    exclude_list = [
        "/",
        "Gym",
        "Reacher-v2",
        "Pusher-v2",
        "InvertedPendulum-v2",
        "InvertedDoublePendulum-v2",
        "HalfCheetah-v2",
        "HalfCheetah-v3",
        "Hopper-v2",
        "Hopper-v3",
        "Walker2d-v2",
        "Walker2d-v3",
        "Swimmer-v2",
        "Swimmer-v3",
        "Ant-v2",
        "Ant-v3",
        "Humanoid-v2",
        "Humanoid-v3",
        "HumanoidStandup-v2",
    ]
    return [
        env_spec
        for env_spec in gym.envs.registration.registry
        if not any(exclude in env_spec for exclude in exclude_list)
    ]

    # return ["CartPole-v1", "MountainCar-v0", "MsPacman-v0"]  # Mock example


def test_model():
    """Test a model based on the user's selections."""

    st.header("Test Model")
    # load an agent
    load_agent()
    if "loaded_agent" in st.session_state:
        # set number of episodes to test agent for
        num_episodes = st.number_input("Number of Episodes", min_value=1, value=10)
        # Render options
        render_option = st.checkbox("Render Test Episodes")
        render_freq = None
        if render_option:
            render_freq = st.number_input(
                "Render Frequency (every 'n' episodes)", min_value=1, value=10
            )

        # Step 4: Test the Model
        if st.button("Start Test"):
            # Step 3: Add Streamlit Callback to the Agent
            progress_bar = st.progress(0)
            terminal_output = st.empty()
            streamlit_callback = stsp.StreamlitCallback(
                progress_bar, terminal_output, num_episodes
            )
            st.session_state.loaded_agent.callback_list.append(streamlit_callback)
            st.session_state.loaded_agent.test(num_episodes, render_option, render_freq)


def setup_sweeps():
    """Setup hyperparameter sweeps for the user's selections."""

    st.header("Setup Hyperparameter Sweeps")

    # Sweep Method Selection
    method = st.radio("Select Sweep Method", ["grid", "random", "bayes"])
    st.write(f"Description: {get_method_description(method)}")
    st.session_state.method = method

    # Project Selection
    projects = [project.name for project in wandb_support.get_projects()]
    project = st.selectbox("Select Project", projects)
    st.session_state.project = project

    # Sweep Name
    name = st.text_input("Enter Sweep Name")
    st.session_state.name = name

    # Number of Sweeps and Episodes
    num_sweeps = st.number_input("Number of Sweeps", min_value=1, value=10)
    st.session_state.num_sweeps = num_sweeps
    num_episodes = st.number_input(
        "Number of Episodes per Sweep", min_value=1, value=100
    )
    st.session_state.num_episodes = num_episodes

    # environment selection
    env = st.selectbox("Select Environment", get_all_gym_envs())
    st.session_state.env = env

    # Save Directory
    if st.button("Select Save Directory"):
        save_dir = select_folder()
        st.session_state.save_dir = save_dir
        if save_dir:
            st.success(f"Save directory set to {save_dir}")

    # Metric Configuration
    goal = st.selectbox("Select Goal", ["maximize", "minimize"])
    st.session_state.goal = goal
    metric_name = st.selectbox(
        "Select Metric", ["episode_reward", "value_loss", "policy_loss"]
    )
    st.session_state.metric_name = metric_name

    # Model Selection
    model_selection = st.multiselect("Select Models", ["Actor Critic", "Reinforce"])
    st.session_state.model_selection = model_selection

    # get config for each model type in model selection
    for model_type in model_selection:
        get_sweep_config_input(model_type)

    # Button to start sweeps
    if st.button("Start Sweeps"):
        for model_type in model_selection:
            # create sweep config
            sweep_config = get_sweep_config(model_type)
            # get agent class
            rl_agent = rl_agents.get_agent_class_from_type(model_type)
            wandb_support.hyperparameter_sweep(
                rl_agent,
                sweep_config,
                st.session_state.num_sweeps,
                st.session_state.num_episodes,
                st.session_state.save_dir,
            )


def get_sweep_config_input(agent_type):
    """Get the hyperparameter sweep configuration for the specified agent type.

    Args:
        agent_type (str): The type of agent to get the configuration for.
    """

    st.header(f"Hyperparameters for {agent_type} agent")
    # Common parameters for both models
    learning_rate_max = st.number_input(
        "Maximum Learning Rate",
        format="%.6f",
        step=1e-6,
        value=1e-3,
        key=f"learning_rate_max_{agent_type}",
        help="The learning rate controls how much to adjust the model's weights with respect to the gradient loss at each iteration. A smaller value may lead to more precise convergence but can slow down the training process.",
    )
    learning_rate_min = st.number_input(
        "Minimum Learning Rate",
        format="%.6f",
        step=1e-6,
        value=1e-5,
        key=f"learning_rate_min_{agent_type}",
        help="The learning rate controls how much to adjust the model's weights with respect to the gradient loss at each iteration. A smaller value may lead to more precise convergence but can slow down the training process.",
    )
    discount_max = st.number_input(
        "Maximum Discount Factor",
        min_value=0.0,
        max_value=1.0,
        value=0.99,
        step=0.01,
        key=f"discount_max_{agent_type}",
        help="The discount factor determines the importance of future rewards. A value of 0 will make the agent short-sighted by only considering current rewards, while a value closer to 1 will make it strive for long-term high rewards.",
    )
    discount_min = st.number_input(
        "Minimum Discount Factor",
        min_value=0.0,
        max_value=1.0,
        value=0.01,
        step=0.01,
        key=f"discount_min_{agent_type}",
        help="The discount factor determines the importance of future rewards. A value of 0 will make the agent short-sighted by only considering current rewards, while a value closer to 1 will make it strive for long-term high rewards.",
    )

    # Policy Model Configuration
    st.subheader(
        "Policy Model Configuration",
        help="The Policy Model network estimates the probabilities of taking actions in a given state.",
    )
    min_policy_num_layers = st.number_input(
        "Policy Model - Minimum Number of Layers",
        min_value=1,
        max_value=10,
        value=1,
        key=f"min_policy_num_layers_{agent_type}",
        help="This sets the depth of the neural network. More layers can capture complex features but may increase the risk of overfitting and require more data and training time.",
    )
    max_policy_num_layers = st.number_input(
        "Policy Model - Maximum Number of Layers",
        min_value=1,
        max_value=10,
        value=2,
        key=f"max_policy_num_layers_{agent_type}",
        help="This sets the depth of the neural network. More layers can capture complex features but may increase the risk of overfitting and require more data and training time.",
    )
    for i in range(1, max_policy_num_layers + 1):
        units = st.multiselect(
            f"Policy Model - Units in Layer {i}",
            [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            key=f"policy_units_in_layer_{i}_{agent_type}",
            help=f"Specify the number of neurons in layer {i} of the Policy Model. More neurons can represent more complex functions but may also lead to overfitting.",
        )
    policy_activation = st.multiselect(
        "Policy Activation Function",
        ["relu", "tanh", "sigmoid"],
        key=f"policy_activation_{agent_type}",
        help="An activation function defines how the neurons in a neural network transform input signals into output signals, adding non-linearity to the learning process. Choose the activation function for the neurons. 'Relu' outputs the input directly if it's positive, otherwise, it outputs zero, and is generally a good default, 'tanh' centers the output between -1 and 1, and 'sigmoid' squashes the output to be between 0 and 1.",
    )
    policy_optimizer = st.multiselect(
        "Policy Optimizer",
        ["adam", "sgd", "rmsprop"],
        key=f"policy_optimizer_{agent_type}",
        help="An optimizer is an algorithm that adjusts the weights of the network to minimize the loss function, guiding how the model learns from the data. Select the optimization algorithm for the policy network. Adam is a popular choice that combines the benefits of two other extensions of stochastic gradient descent.",
    )

    # Value Model Configuration
    st.subheader(
        "Value Model Configuration",
        help=" The Value Model network estimates the value of being in a given state.",
    )
    min_value_num_layers = st.number_input(
        "Value Model - Minimum Number of Layers",
        min_value=1,
        max_value=10,
        value=1,
        key=f"min_value_num_layers_{agent_type}",
        help="This sets the depth of the neural network. Maximum More layers can capture complex features but may increase the risk of overfitting and require more data and training time.",
    )
    max_value_num_layers = st.number_input(
        "Value Model - Number of Layers",
        min_value=1,
        max_value=10,
        value=2,
        key=f"max_value_num_layers_{agent_type}",
        help="This sets the depth of the neural network. More layers can capture complex features but may increase the risk of overfitting and require more data and training time.",
    )
    for i in range(1, max_value_num_layers + 1):
        units = st.multiselect(
            f"Value Model - Units in Layer {i}",
            [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            key=f"value_units_in_layer_{i}_{agent_type}",
            help=f"Specify the number of neurons in layer {i} of the Policy Model. More neurons can represent more complex functions but may also lead to overfitting.",
        )
    value_activation = st.multiselect(
        "Value Activation Function",
        ["relu", "tanh", "sigmoid"],
        key=f"value_activation_{agent_type}",
        help="An activation function defines how the neurons in a neural network transform input signals into output signals, adding non-linearity to the learning process. Choose the activation function for the neurons. 'Relu' outputs the input directly if it's positive, otherwise, it outputs zero, and is generally a good default, 'tanh' centers the output between -1 and 1, and 'sigmoid' squashes the output to be between 0 and 1.",
    )
    value_optimizer = st.multiselect(
        "Value Optimizer",
        ["adam", "sgd", "rmsprop"],
        key=f"value_optimizer_{agent_type}",
        help="An optimizer is an algorithm that adjusts the weights of the network to minimize the loss function, guiding how the model learns from the data. Select the optimization algorithm for the policy network. Adam is a popular choice that combines the benefits of two other extensions of stochastic gradient descent.",
    )

    # Specific parameters for Actor Critic
    if agent_type == "Actor Critic":
        st.subheader("Actor Critic Configuration")
        max_policy_trace_decay = st.number_input(
            "Maximum Policy Trace Decay",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.01,
            key=f"max_policy_trace_decay_{agent_type}",
            help="This parameter controls the decay rate of the eligibility traces for the policy model. It's a factor that determines how quickly the influence of past actions decreases over time. A higher decay rate means past actions remain influential for longer, leading to smoother policy updates but potentially slower convergence. A lower decay rate makes the policy more reactive to recent actions but may increase the risk of instability. Typically set between 0 (no trace, more reactive) and 1 (long trace, smoother updates). Adjust this to balance between exploration efficiency and policy stability.",
        )
        min_policy_trace_decay = st.number_input(
            "Minimum Policy Trace Decay",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            key=f"min_policy_trace_decay_{agent_type}",
            help="This parameter controls the decay rate of the eligibility traces for the policy model. It's a factor that determines how quickly the influence of past actions decreases over time. A higher decay rate means past actions remain influential for longer, leading to smoother policy updates but potentially slower convergence. A lower decay rate makes the policy more reactive to recent actions but may increase the risk of instability. Typically set between 0 (no trace, more reactive) and 1 (long trace, smoother updates). Adjust this to balance between exploration efficiency and policy stability.",
        )
        max_value_trace_decay = st.number_input(
            "Maximum Value Trace Decay",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.01,
            key=f"max_value_trace_decay_{agent_type}",
            help="This parameter sets the decay rate for the eligibility traces in the value model. It's crucial for temporal-difference learning, influencing how future rewards are attributed to past states. A higher decay rate retains the influence of earlier states for longer periods, which can be beneficial in environments where delayed rewards are significant. On the other hand, a lower decay rate focuses more on recent states, potentially leading to quicker, but less stable learning. Generally set between 0 and 1, where 0 focuses solely on the most recent state and 1 considers a longer history of states for value updates.",
        )
        min_value_trace_decay = st.number_input(
            "Minimum Value Trace Decay",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            key=f"min_value_trace_decay_{agent_type}",
            help="This parameter sets the decay rate for the eligibility traces in the value model. It's crucial for temporal-difference learning, influencing how future rewards are attributed to past states. A higher decay rate retains the influence of earlier states for longer periods, which can be beneficial in environments where delayed rewards are significant. On the other hand, a lower decay rate focuses more on recent states, potentially leading to quicker, but less stable learning. Generally set between 0 and 1, where 0 focuses solely on the most recent state and 1 considers a longer history of states for value updates.",
        )


def get_sweep_config(agent_type):
    """Returns the sweep configuration for the specified agent type.

    Args:
        agent_type (str): The type of agent to get the configuration for.

    Returns:
        A dictionary containing the sweep configuration.
    """

    sweep_config = {
        "method": st.session_state.method,
        "project": st.session_state.project,
        "name": st.session_state.name,
        "metric": {"name": st.session_state.metric_name, "goal": st.session_state.goal},
        "parameters": {
            "learning_rate": {
                "max": st.session_state[f"learning_rate_max_{agent_type}"],
                "min": st.session_state[f"learning_rate_min_{agent_type}"],
            },
            "discount": {
                "max": st.session_state[f"discount_max_{agent_type}"],
                "min": st.session_state[f"discount_min_{agent_type}"],
            },
            "policy_num_layers": {
                "max": st.session_state[f"max_policy_num_layers_{agent_type}"],
                "min": st.session_state[f"min_policy_num_layers_{agent_type}"],
            },
            "policy_activation": {
                "values": st.session_state[f"policy_activation_{agent_type}"]
            },
            "policy_optimizer": {
                "values": st.session_state[f"policy_optimizer_{agent_type}"]
            },
            "value_num_layers": {
                "max": st.session_state[f"max_value_num_layers_{agent_type}"],
                "min": st.session_state[f"min_value_num_layers_{agent_type}"],
            },
            "value_activation": {
                "values": st.session_state[f"value_activation_{agent_type}"]
            },
            "value_optimizer": {
                "values": st.session_state[f"value_optimizer_{agent_type}"]
            },
            "env": {"value": st.session_state.env},
            "model_type": {"value": agent_type},
        },
    }

    for i in range(1, st.session_state[f"max_policy_num_layers_{agent_type}"] + 1):
        sweep_config["parameters"][f"policy_units_layer_{i}"] = {
            "values": st.session_state[f"policy_units_in_layer_{i}_{agent_type}"]
        }
    for i in range(1, st.session_state[f"max_value_num_layers_{agent_type}"] + 1):
        sweep_config["parameters"][f"value_units_layer_{i}"] = {
            "values": st.session_state[f"value_units_in_layer_{i}_{agent_type}"]
        }

    if agent_type == "Actor Critic":
        sweep_config["parameters"]["policy_trace_decay"] = {
            "max": st.session_state[f"max_policy_trace_decay_{agent_type}"],
            "min": st.session_state[f"min_policy_trace_decay_{agent_type}"],
        }
        sweep_config["parameters"]["value_trace_decay"] = {
            "max": st.session_state[f"max_value_trace_decay_{agent_type}"],
            "min": st.session_state[f"min_value_trace_decay_{agent_type}"],
        }

    return sweep_config


def get_method_description(method):
    """Returns the description for the specified sweep method.

    Args:
        method (str): The name of the sweep method.

    Returns:
        A string containing the description for the specified sweep method.
    """

    descriptions = {
        "grid": "Iterate over every combination of hyperparameter values.",
        "random": "Choose a random set of hyperparameter values based on distributions.",
        "bayes": "Use Gaussian Process to model the relationship between parameters and metric.",
    }

    return descriptions.get(method, "")


def wandb_utils():
    """A collection of helper functions for interacting with Weights & Biases."""

    # Function to login to W&B using the provided API key
    try:
        login_to_wandb()
    except ValueError:
        st.error("Invalid API key. Please try again.")

    # button to delete all runs from a project
    if st.button("Delete All Runs"):
        wandb_support.delete_all_runs(st.session_state.project)

    # button to flush the wandb cache
    if st.button("Flush Cache"):
        wandb_support.flush_cache()
        st.success("Cache flushed successfully.")


if __name__ == "__main__":
    main()
