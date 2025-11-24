import sys
import os

from pydantic_core.core_schema import str_schema

# Use environment variable for IsaacLab path, with fallback to relative path
ISAACLAB_PATH = os.environ.get('ISAACLAB_PATH', os.path.join(os.path.dirname(__file__), '..', '..', 'IsaacLab', 'source'))
ISAACLAB_TASKS_PATH = os.path.join(ISAACLAB_PATH, 'isaaclab_tasks')

sys.path.append(ISAACLAB_PATH)
sys.path.append(ISAACLAB_TASKS_PATH)

import json
import dataclasses
from typing import Optional, Dict, List
from abc import ABC, abstractmethod
from collections import deque
import numpy as np
import torch as T

import gymnasium as gym
import gymnasium_robotics
from gymnasium.envs.registration import EnvSpec, WrapperSpec
import gymnasium.wrappers as gym_wrappers
import gymnasium.wrappers.vector as gym_vector_wrappers
# from gymnasium.wrappers import (
#     AtariPreprocessing,
#     TimeLimit,
#     TimeAwareObservation,
#     FrameStackObservation,
#     ResizeObservation
# )

from gymnasium.vector import VectorEnv, SyncVectorEnv
# Register gymnasium robotics with gymnasium
# gym.register_envs(gymnasium_robotics)

class NStepReward(gym.Wrapper):
    def __init__(self, env, n, discount=0.99):
        """Initialize the wrapper with the environment and number of steps to track.

        Args:
            env (gym.Env): The Gymnasium environment to wrap.
            n (int): The number of previous steps to include in the trajectory.
            discount (float): The discount factor for the trajectory.
        """
        super().__init__(env)
        self.env = env
        self.n = n
        self.n_states = deque(maxlen=self.n)
        self.n_actions = deque(maxlen=self.n)
        self.n_rewards = deque(maxlen=self.n)
        self.n_next_states = deque(maxlen=self.n)
        self.n_dones = deque(maxlen=self.n)
        self.n_state_achieved_goals = deque(maxlen=self.n)
        self.n_next_state_achieved_goals = deque(maxlen=self.n)
        self.n_desired_goals = deque(maxlen=self.n)
        self.current_state = None
        self.step_count = 0
        # self.rewards = deque(maxlen=self.n)
        self.discount = discount

    def reset(self, **kwargs):
        """Reset the environment and clear the trajectory history.

        Args:
            **kwargs: Additional arguments for env.reset().

        Returns:
            tuple: (observation, info) from the environment reset.
        """
        #DEBUG
        # print(f'n-step trajectory reset called')
        # self.step_count = 0
        # Capture current n-step trajectory info to return in info dict
        trajectory = {
            'states': np.array(self.n_states),
            'actions': np.array(self.n_actions),
            'rewards': np.array(self.n_rewards),
            'next_states': np.array(self.n_next_states),
            'dones': np.array(self.n_dones)
        }
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            trajectory['state_achieved_goals'] = np.array(self.n_state_achieved_goals)
            trajectory['next_state_achieved_goals'] = np.array(self.n_next_state_achieved_goals)
            trajectory['desired_goals'] = np.array(self.n_desired_goals)

        state, info = self.env.reset(**kwargs)
        #DEBUG
        # print(f'n-step trajectory reset state:{state}, info:{info}')

        self.n_states = deque(maxlen=self.n)
        self.n_actions = deque(maxlen=self.n)
        self.n_rewards = deque(maxlen=self.n)
        self.n_next_states = deque(maxlen=self.n)
        self.n_dones = deque(maxlen=self.n)
        # self.rewards.clear()

        action_shape = self.env.action_space.shape
        # Add state achieved, next achieved, and desired goals if state is dict and has attrs
        if isinstance(state, dict):
            state_shape = self.env.observation_space['observation'].shape
            goal_shape = self.env.observation_space['achieved_goal'].shape
            self.n_state_achieved_goals = deque(maxlen=self.n)
            self.n_next_state_achieved_goals = deque(maxlen=self.n)
            self.n_desired_goals = deque(maxlen=self.n)
            for _ in range(self.n):
                self.n_state_achieved_goals.append(np.zeros(goal_shape))
                self.n_next_state_achieved_goals.append(np.zeros(goal_shape))
                self.n_desired_goals.append(np.zeros(goal_shape))
        else:
            state_shape = self.env.observation_space.shape

        for _ in range(self.n):
            self.n_states.append(np.zeros(state_shape))
            self.n_actions.append(np.zeros(action_shape))
            self.n_rewards.append(0)
            self.n_next_states.append(np.zeros(state_shape))
            self.n_dones.append(0)
        
        self.current_state = state
        info['n-step trajectory'] = trajectory
        #DEBUG
        # print(f'n-step trajectory reset info:{info}')
        return state, info

    def step(self, action):
        """Step the environment and update the n-step trajectory.

        Args:
            action: The action to take in the environment.

        Returns:
            tuple: (observation, reward, terminated, truncated, info) with updated info dict.
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        # self.rewards.append(reward)
        # discounts = np.array([self.discount ** i for i in range(len(self.rewards))])
        # rewards = np.array(self.rewards)
        # reward = np.sum(rewards * discounts)
        done = terminated or truncated
        # done = terminated or truncated
        self.step_count += 1
        # If current step == 1, add state, action, and next state to every idx
        if self.step_count == 1:
            for _ in range(self.n):
                if isinstance(self.env.observation_space, gym.spaces.Dict):
                    self.n_states.append(self.current_state['observation'])
                    self.n_actions.append(action)
                    self.n_next_states.append(next_state['observation'])
                    self.n_state_achieved_goals.append(self.current_state['achieved_goal'])
                    self.n_next_state_achieved_goals.append(next_state['achieved_goal'])
                    self.n_desired_goals.append(self.current_state['desired_goal'])
                else:
                    self.n_states.append(self.current_state)
                    self.n_actions.append(action)
                    self.n_next_states.append(next_state)
        else:
            # Append the current step's data to the trajectory
            if isinstance(self.env.observation_space, gym.spaces.Dict):
                self.n_states.append(self.current_state['observation'])
                self.n_actions.append(action)
                self.n_next_states.append(next_state['observation'])
                self.n_state_achieved_goals.append(self.current_state['achieved_goal'])
                self.n_next_state_achieved_goals.append(next_state['achieved_goal'])
                self.n_desired_goals.append(self.current_state['desired_goal'])
            else:
                self.n_states.append(self.current_state)
                self.n_actions.append(action)
                self.n_next_states.append(next_state)
            
        self.n_rewards.append(reward)
        self.n_dones.append(done)

        # Update the current state
        self.current_state = next_state

        # Construct the trajectory dictionary
        trajectory = {
            'states': np.array(self.n_states),
            'actions': np.array(self.n_actions),
            'rewards': np.array(self.n_rewards),
            'next_states': np.array(self.n_next_states),
            'dones': np.array(self.n_dones)
        }
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            trajectory['state_achieved_goals'] = np.array(self.n_state_achieved_goals)
            trajectory['next_state_achieved_goals'] = np.array(self.n_next_state_achieved_goals)
            trajectory['desired_goals'] = np.array(self.n_desired_goals)
        # # Add the trajectory to the info dictionary
        info['n-step trajectory'] = trajectory
        #DEBUG
        # print(f'n-step trajectory step info:{info}')
        return next_state, reward, terminated, truncated, info

    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space

    @property
    def single_action_space(self):
        return self.env.single_action_space

    @property
    def single_observation_space(self):
        return self.env.single_observation_space

class VectorNStepReward(gym.Wrapper):
    def __init__(self, env: VectorEnv, n: int, goal_aware: bool = False,
                 obs_key: str = 'observation', goal_key: str | None = None, ach_goal_key: str | None = None):
        """
        Initialize the vectorized wrapper for n-step trajectories.
        Args:
            env (gym.VectorEnv | ManagerBasedRLEnv): The vectorized environment to wrap.
            n (int): The number of previous steps to include in the trajectory.
            goal_aware (bool): Is the observation space goal-aware?
            obs_key (str): The key for the observation space.
            goal_key (str | None): The key for the goal space.
            ach_goal_key (str | None): The key for the achieved goal space.
        """
        super().__init__(env)
        self.n = n
        self.num_envs = env.num_envs if hasattr(env, 'num_envs') else 1
        self.goal_aware = goal_aware
        self.obs_key = obs_key
        self.goal_key = goal_key
        self.ach_goal_key = ach_goal_key

        # Per env deques for trajectories
        self.n_states = [deque(maxlen=self.n) for _ in range(self.num_envs)]
        self.n_actions = [deque(maxlen=self.n) for _ in range(self.num_envs)]
        self.n_rewards = [deque(maxlen=self.n) for _ in range(self.num_envs)]
        self.n_next_states = [deque(maxlen=self.n) for _ in range(self.num_envs)]
        self.n_dones = [deque(maxlen=self.n) for _ in range(self.num_envs)]

        if self.goal_aware:
            self.n_state_achieved_goals = [deque(maxlen=self.n) for _ in range(self.num_envs)]
            self.n_next_state_achieved_goals = [deque(maxlen=self.n) for _ in range(self.num_envs)]
            self.n_desired_goals = [deque(maxlen=self.n) for _ in range(self.num_envs)]

        # Initialize trajectory buffers
        self._initialize_trajectory_buffers()

        self.current_states = None
        self.step_counts = T.zeros(self.num_envs, dtype=T.int32)

    def reset(self, **kwargs):
        """
        Reset all envs and clear per-env trajectories.
        Returns batched (observations, infos)
        """
        states, infos = self.env.reset(**kwargs)

        # Convert states to numpy arrays
        # if isinstance(states, T.Tensor):
        #     states = states.cpu().numpy()

        # Capture existing trajectories
        trajectory = self._build_trajectory()

        # Clear env deques
        for i in range(self.num_envs):
            self.n_states[i].clear()
            self.n_actions[i].clear()
            self.n_rewards[i].clear()
            self.n_next_states[i].clear()
            self.n_dones[i].clear()
            if self.goal_aware:
                self.n_state_achieved_goals[i].clear()
                self.n_next_state_achieved_goals[i].clear()
                self.n_desired_goals[i].clear()

        # Initialize trajectory buffers (reset to zeros)
        self._initialize_trajectory_buffers()

        self.current_states = states

        if 'n-step trajectory' not in infos:
            infos['n-step trajectory'] = {}
        infos['n-step trajectory'].update(trajectory)

        return states, infos
    
    def step(self, actions: T.Tensor):
        """
        Step all envs with batched actions, update per-env trajectories.
        Returns batched (next_states, rewards, dones, infos)
        """
        # Convert to numpy if tensors
        # if isinstance(actions, T.Tensor):
        #     actions = actions.cpu().numpy()

        next_states, rewards, terminations, truncations, infos = self.env.step(actions)
        dones = terminations | truncations
        self.step_counts += 1

        for i in range(self.num_envs):
            state = self.current_states[self.obs_key][i] if self.obs_key is not None else self.current_states[i]
            next_state = next_states[self.obs_key][i] if self.obs_key is not None else next_states[i]

            if self.step_counts[i] == 1:
                # Bootstrap first step across n
                for _ in range(self.n):
                    self.n_states[i].append(state)
                    self.n_actions[i].append(actions[i])
                    self.n_rewards[i].append(rewards[i])
                    self.n_next_states[i].append(next_state)
                    self.n_dones[i].append(dones[i])
                    if self.goal_aware:
                        self.n_state_achieved_goals[i].append(self.current_states[self.ach_goal_key][i] if self.ach_goal_key is not None else None)
                        self.n_next_state_achieved_goals[i].append(next_states[self.ach_goal_key][i] if self.ach_goal_key is not None else None)
                        self.n_desired_goals[i].append(self.current_states[self.goal_key][i] if self.goal_key is not None else None)

            else:
                # Append current step
                self.n_states[i].append(state)
                self.n_actions[i].append(actions[i])
                self.n_rewards[i].append(rewards[i])
                self.n_next_states[i].append(next_state)
                self.n_dones[i].append(dones[i])
                if self.goal_aware:
                    self.n_state_achieved_goals[i].append(self.current_states[self.ach_goal_key][i] if self.ach_goal_key is not None else None)
                    self.n_next_state_achieved_goals[i].append(next_states[self.ach_goal_key][i] if self.ach_goal_key is not None else None)
                    self.n_desired_goals[i].append(self.current_states[self.goal_key][i] if self.goal_key is not None else None)

        self.current_states = next_states

        # Build batched trajectory
        trajectory = self._build_trajectory()
        infos['n-step trajectory'] = trajectory

        return next_states, rewards, dones, infos

    def _build_trajectory(self):
        """Construct batched n-step trajectory dict from per-env deques."""
        print(f'self.n_states:{self.n_states}')
        print(f'self.n_actions:{self.n_actions}')
        print(f'self.n_rewards:{self.n_rewards}')
        print(f'self.n_next_states:{self.n_next_states}')
        print(f'self.n_dones:{self.n_dones}')

        trajectory = {
            'states': T.stack([T.stack(list(d)) for d in self.n_states]),
            'actions': T.stack([T.stack(list(d)) for d in self.n_actions]),
            'rewards': T.stack([T.stack(list(d)) for d in self.n_rewards]),
            'next_states': T.stack([T.stack(list(d)) for d in self.n_next_states]),
            'dones': T.stack([T.stack(list(d)) for d in self.n_dones])
        }
        if self.goal_aware:
            trajectory['state_achieved_goals'] = T.stack([T.stack(list(d)) for d in self.n_state_achieved_goals])
            trajectory['next_state_achieved_goals'] = T.stack([T.stack(list(d)) for d in self.n_next_state_achieved_goals])
            trajectory['desired_goals'] = T.stack([T.stack(list(d)) for d in self.n_desired_goals])
        
        return trajectory

    def _initialize_trajectory_buffers(self):
        """Initialize trajectory deques with zeros."""
        # Get shapes from single spaces
        single_obs_space = self.env.single_observation_space if hasattr(self.env, 'single_observation_space') else self.env.observation_space
        single_act_space = self.env.single_action_space if hasattr(self.env, 'single_action_space') else self.env.action_space
        state_shape = single_obs_space[self.obs_key].shape if self.obs_key is not None else single_obs_space.shape
        action_shape = single_act_space.shape
        goal_shape = single_obs_space[self.goal_key].shape if self.goal_aware and self.goal_key is not None else None

        # Initialize deques with zeros per env
        for i in range(self.num_envs):
            for _ in range(self.n):
                self.n_states[i].append(T.zeros(state_shape))
                self.n_actions[i].append(T.zeros(action_shape))
                self.n_rewards[i].append(T.zeros(1))
                self.n_next_states[i].append(T.zeros(state_shape))
                self.n_dones[i].append(T.zeros(1))
                if self.goal_aware:
                    self.n_state_achieved_goals[i].append(T.zeros(goal_shape))
                    self.n_next_state_achieved_goals[i].append(T.zeros(goal_shape))
                    self.n_desired_goals[i].append(T.zeros(goal_shape))

    @property
    def single_action_space(self):
        return self.env.single_action_space

    @property
    def single_observation_space(self):
        return self.env.single_observation_space


WRAPPER_REGISTRY = {
    "AtariPreprocessing": {
        "cls": gym_wrappers.AtariPreprocessing,
        "default_params": {
            "frame_skip": 1,
            "grayscale_obs": True,
            "scale_obs": True
        }
    },
    "TimeLimit": {
        "cls": gym_wrappers.TimeLimit,
        "default_params": {
            "max_episode_steps": 1000
        }
    },
    "TimeAwareObservation": {
        "cls": gym_wrappers.TimeAwareObservation,
        "default_params": {
            "flatten": False,
            "normalize_time": False
        }
    },
    "FrameStackObservation": {
        "cls": gym_wrappers.FrameStackObservation,
        "default_params": {
            "stack_size": 4
        }
    },
    "ResizeObservation": {
        "cls": gym_wrappers.ResizeObservation,
        "default_params": {
            "shape": 84
        }
    },
    "NStepReward": {
        "cls": NStepReward,
        "default_params": {"n": 1}
    },
    "VectorNStepReward": {
        "cls": VectorNStepReward,
        "default_params": {"n": 1, "obs_key": 'observation', "goal_key": None, "ach_goal_key": None}
    }
}

# def atari_wrappers(env):
#     """
#     Wrap an Atari environment with preprocessing and frame stacking.

#     This function applies standard Atari preprocessing, including converting to grayscale,
#     resizing, scaling, and stacking multiple consecutive frames for better temporal
#     context.

#     Args:
#         env (gym.Env): The original Atari environment.

#     Returns:
#         gym.Env: The wrapped environment with preprocessing and frame stacking applied.
#     """
#     env = AtariPreprocessing(
#         env,
#         frame_skip=1,
#         grayscale_obs=True,
#         scale_obs=True,
#         screen_size=84
#     )
#     env = FrameStackObservation(env, stack_size=4)
#     return env

def wrap_env(vec_env, wrappers):
    wrapper_list = []
    for wrapper in wrappers:
        if wrapper['type'] in WRAPPER_REGISTRY:
            # print(f'wrapper type:{wrapper["type"]}')
            # Use a copy of default_params to avoid modifying the registry
            default_params = WRAPPER_REGISTRY[wrapper['type']]["default_params"].copy()
            
            if wrapper['type'] == "ResizeObservation":
                # Ensure shape is a tuple for ResizeObservation
                default_params['shape'] = (default_params['shape'], default_params['shape']) if isinstance(default_params['shape'], int) else default_params['shape']
            
            # print(f'default params:{default_params}')
            override_params = wrapper.get("params", {})
            
            if wrapper['type'] == "ResizeObservation":
                # Ensure override_params shape is a tuple
                if 'shape' in override_params:
                    override_params['shape'] = (override_params['shape'], override_params['shape']) if isinstance(override_params['shape'], int) else override_params['shape']
            
            # print(f'override params:{override_params}')
            final_params = {**default_params, **override_params}
            # print(f'final params:{final_params}')
            
            def wrapper_factory(env, cls=WRAPPER_REGISTRY[wrapper['type']]["cls"], params=final_params):
                return cls(env, **params)
            
            wrapper_list.append(wrapper_factory)
    
    # Define apply_wrappers outside the loop
    def apply_wrappers(env):
        for wrapper in wrapper_list:
            env = wrapper(env)
            # print(f'length of obs space:{len(env.observation_space.shape)}')
            # print(f'env obs space shape:{env.observation_space.shape}')
        return env
    
    # print(f'wrapper list:{wrapper_list}')
    envs = [lambda: apply_wrappers(gym.make(vec_env.spec.id, render_mode="rgb_array")) for _ in range(vec_env.num_envs)]    
    return SyncVectorEnv(envs)

class EnvWrapper(ABC):
    """
    Abstract base class for environment wrappers.

    This class defines the required interface for custom environment wrappers.
    """

    @abstractmethod
    def reset(self):
        """
        Reset the environment to an initial state.

        Returns:
            Any: Initial observation of the environment.
        """
        pass
    
    @abstractmethod
    def step(self, action):
        """
        Take an action in the environment.

        Args:
            action: The action to be taken.

        Returns:
            Tuple: Observation, reward, done flag, and additional info.
        """
        pass

    @abstractmethod
    def _initialize_env(self, num_envs: int = 1, seed: Optional[int] = None, render_mode: Optional[str] = None):
        """
        Initialize the environment with optional rendering and seeding.

        Args:
            num_envs (int): Number of parallel environments (default: 1).
            seed (int): Random seed for the environment (default: None).
            render_mode (Optional[str]): Render mode for the environment (default: None).

        Returns:
            Any: The initialized environment.
        """
        pass

    def clone(self, num_envs: int = 1, seed: Optional[int] = None, render_mode: Optional[str] = None) -> 'EnvWrapper':
        """
        Create a new instance of the environment wrapper with the passed parameters.

        Args:
            num_envs (int): Number of parallel environments (default: 1).
            seed (Optional[int]): Seed for the environment. If None, a random seed is used. (default: None).
            render_mode (Optional[str]): Render mode for the environment (default: None).

        Returns:
            EnvWrapper: A new instance of the environment wrapper.
        """
        json_config = self.to_json()
        clone = self.from_json(json_config)
        # clone.env = clone._initialize_env(num_envs, seed, render_mode)
        return clone

    @abstractmethod
    def format_actions(self, actions: np.ndarray | T.Tensor, testing: bool = False):
        """
        Format actions for the environment.

        Args:
            actions: Actions to format.
            testing (bool): Whether in testing mode (default: False).

        Returns:
            Any: Formatted actions.
        """
        pass
    
    @property
    @abstractmethod
    def observation_space(self):
        """
        Get the observation space of the environment.

        Returns:
            gym.Space: The observation space.
        """
        pass
    
    @property
    @abstractmethod
    def action_space(self):
        """
        Get the action space of the environment.

        Returns:
            gym.Space: The action space.
        """
        pass

    @property
    def single_action_space(self):
        """
        Get the single action space for vectorized environments.

        Returns:
            gym.Space: The single action space.
        """
        pass

    @property
    def single_observation_space(self):
        """
        Get the single observation space for vectorized environments.

        Returns:
            gym.Space: The single observation space.
        """
        pass

    @abstractmethod
    def to_json(self) -> str:
        """
        Serialize the environment wrapper configuration to JSON.

        Returns:
            str: JSON string representing the environment configuration.
        """
        pass

    @classmethod
    def from_json(cls, json_string: str):
        """
        Create an environment wrapper instance from a JSON string.

        This method will delegate to the appropriate subclass's `from_json` method
        based on the type specified in the JSON.

        Args:
            json_string (str): JSON string representing the environment configuration.

        Returns:
            EnvWrapper: A new environment wrapper instance.

        Raises:
            ValueError: If the type in the JSON is not recognized or if instantiation fails.
        """
        config = json.loads(json_string)
        try:
            if config['type'] == 'GymnasiumWrapper':
                return GymnasiumWrapper.from_json(json_string)
            # Add more conditions here for other subclasses if they exist
            elif config['type'] == 'IsaacSimWrapper':
                return IsaacSimWrapper.from_json(json_string)
            else:
                raise ValueError(f"Unknown environment wrapper type: {config['type']}")
        except KeyError as e:
            raise ValueError(f"Missing 'type' key in JSON configuration: {e}")
        except Exception as e:
            raise ValueError(f"Failed to instantiate environment from JSON: {e}")


class GymnasiumWrapper(EnvWrapper):
    """
    Wrapper for Gymnasium environments with additional utilities.

    This wrapper supports initialization, resetting, stepping, rendering,
    and JSON-based serialization of Gymnasium environments.
    """
    def __init__(self, env_spec: EnvSpec, wrappers: Optional[list[dict]] = None, num_envs: int = 1,
                 seed: Optional[int] = None, render_mode: Optional[str] = None):
        self.env_spec = env_spec
        self.wrappers = wrappers
        self.num_envs = 1
        if seed is None:
            seed = np.random.randint(1000)
        self.seed = seed
        self.render_mode = render_mode
        self.env = self._initialize_env()
        

    def _initialize_env(self):
        """
        Initialize the Gymnasium environment with unique seeds for each environment.

        
        Returns:
            gym.VectorEnv: The initialized Gymnasium vectorized environment.
        """
        single_wrappers = []
        vector_wrappers = []
        if self.wrappers:
            for wrapper in self.wrappers:
                wrapper_type = wrapper.get('type')
                if not wrapper_type:
                    raise ValueError("Each wrapper dict must have a 'type' key.")
                
                if wrapper_type in WRAPPER_REGISTRY:
                    entry = WRAPPER_REGISTRY[wrapper_type]
                    cls = entry["cls"]
                    default_params = entry["default_params"].copy()
                    vector_aware = entry.get("vector_aware", False)
                else:
                    # Dynamic resolution for built-in Gymnasium wrappers
                    if hasattr(gym_vector_wrappers, wrapper_type):
                        cls = getattr(gym_vector_wrappers, wrapper_type)
                        vector_aware = True
                    elif hasattr(gym_wrappers, wrapper_type):
                        cls = getattr(gym_wrappers, wrapper_type)
                        vector_aware = False
                    else:
                        raise ValueError(f"Unknown wrapper type '{wrapper_type}'. Add to WRAPPER_REGISTRY or ensure it's a valid Gymnasium wrapper class name.")
                    
                    default_params = {}  # No defaults for unresolved; rely on user params
                
                override_params = wrapper.get("params", {})
                final_params = {**default_params, **override_params}
                
                if vector_aware:
                    vector_wrappers.append((cls, final_params))
                else:
                    def wrapper_fn(env, cls=cls, params=final_params):
                        return cls(env, **params)
                    single_wrappers.append(wrapper_fn)

        # Create vector env with single-env wrappers applied per sub-env
        vec_env = gym.make_vec(
            id=self.env_spec,
            num_envs=self.num_envs,
            vectorization_mode="sync",
            wrappers=single_wrappers,
            render_mode=self.render_mode
        )

        # # Manually seed each sub-env
        # for i, sub_env in enumerate(vec_env.envs):
        #     sub_env.action_space.seed(seeds[i])
        #     if hasattr(sub_env, 'seed'):
        #         sub_env.seed(seeds[i])
        #     if hasattr(sub_env.observation_space, 'seed'):
        #         sub_env.observation_space.seed(seeds[i])

        # Apply vector-aware wrappers to the entire vec_env
        for cls, params in vector_wrappers:
            vec_env = cls(vec_env, **params)

        return vec_env
        

    def reset(self):
        #DEBUG
        # print(f'GymnasiumWrapper reset called')
        if self.seed is not None:
            state, info = self.env.reset(seed=self.seed)
        else:
            state, info = self.env.reset()
        #DEBUG
        # print(f'GymnasiumWrapper reset state:{state}, info:{info}')
        return state, info

    def step(self, action):
        states, rewards, terms, truncs, infos = self.env.step(action)
        dones = np.logical_or(terms, truncs)
        
        return states, rewards, dones, infos
    
    def format_actions(self, actions: np.ndarray | T.Tensor):
        if isinstance(actions, T.Tensor):
            actions = actions.cpu().numpy()
        if isinstance(self.action_space, gym.spaces.Box):
            # if testing:
            #     num_envs = 1
            # else:
            num_envs = self.env.num_envs
            num_actions = self.action_space.shape[-1]
            return actions.reshape(num_envs, num_actions)
        if isinstance(self.action_space, gym.spaces.Discrete) or isinstance(self.action_space, gym.spaces.MultiDiscrete):
            return actions.ravel()
        
    def get_base_env(self, env_idx:int=0):
        """Recursively unwrap an environment to get the base environment."""
        env = self.env.envs[env_idx]
        while hasattr(env, 'env'):
            env = env.env
        return env
    
    def close(self):
        """
        Close the environment.
        """
        self.env.close()
    
    @property
    def observation_space(self):
        """
        Get the observation space of the environment.

        Returns:
            gym.Space: The observation space.
        """
        return self.env.observation_space
    
    @property
    def action_space(self):
        """
        Get the action space of the environment.

        Returns:
            gym.Space: The action space.
        """
        return self.env.action_space
    
    @property
    def single_action_space(self):
        """
        Get the single action space for vectorized environments.

        Returns:
            gym.Space: The single action space.
        """
        return self.env.single_action_space

    @property
    def single_observation_space(self):
        """
        Get the single observation space for vectorized environments.

        Returns:
            gym.Space: The single observation space.
        """
        return self.env.single_observation_space
    
    @property
    def config(self):
        """
        Get the configuration of the wrapper.

        Returns:
            dict: Configuration dictionary.
        """
        return {
            "type": self.__class__.__name__,
            "env": self.env_spec.to_json(),
            "wrappers": self.wrappers
        }
    
    def to_json(self):
        """
        Serialize the wrapper configuration to JSON.

        Returns:
            str: JSON string representing the configuration.
        """
        return json.dumps(self.config)

    @classmethod
    def from_json(cls, json_env_spec):
        """
        Create a Gymnasium wrapper instance from a JSON string.

        Args:
            json_env_spec (str): JSON string representing the configuration.

        Returns:
            GymnasiumWrapper: A new Gymnasium wrapper instance.
        """
        config = json.loads(json_env_spec)
        env_spec = EnvSpec.from_json(config['env'])
        try:
            return cls(env_spec, config["wrappers"])
        except Exception as e:
            raise ValueError(f"Environment wrapper error: {config}, {e}")
    
class IsaacSimWrapper(EnvWrapper):
    def __init__(self, cfg: str, num_envs: int = 1, wrappers: Optional[list[dict]] = None, render_mode: Optional[str] = 'headless',
                 seed: Optional[int] = None, obs_key: str = 'observation', goal_key: str | None = None):
        """
        Placeholder wrapper for Isaac Sim environments.

        This class is a template and needs implementation based on Isaac Sim's API.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.wrappers = wrappers
        self.render_mode = render_mode
        self.obs_key = obs_key
        self.goal_key = goal_key
        if seed is None:
            seed = np.random.randint(1000)
        self.seed = seed
        self.env = self._initialize_env()

    def _initialize_env(self):
        """
        Initialize the Isaac Sim environment with unique seeds for each environment.
        """
        import importlib

        # Initialize Omniverse app FIRST - this is critical for Isaac Lab
        # The app must be running before importing ManagerBasedRLEnv
        try:
            import omni.kit.app as kit_app
            self.app = kit_app.get_app()
        except Exception:
            self.app = None
        if self.app is None:
            from isaaclab.app import AppLauncher
            app_launcher = AppLauncher(headless=(self.render_mode=='headless'), device="cuda:0")
            self.app = app_launcher.app
        
        from isaaclab.envs import ManagerBasedRLEnv

        module_path, class_name = self.cfg.split(':')
        cfg_class = getattr(importlib.import_module(module_path), class_name)
        cfg = cfg_class()
        #DEBUG
        # print(f'IsaacSimWrapper initialize_env cfg: {cfg}')
        cfg.scene.num_envs = self.num_envs
        cfg.sim.device = "cuda:0"
        cfg.seed = self.seed

        env = ManagerBasedRLEnv(cfg=cfg)
        if self.wrappers:
            for wrapper in self.wrappers:
                if wrapper['type'] in WRAPPER_REGISTRY:
                    default_params = WRAPPER_REGISTRY[wrapper['type']]["default_params"].copy()
                    override_params = wrapper.get("params", {})
                    final_params = {**default_params, **override_params}
                    env = WRAPPER_REGISTRY[wrapper['type']]["cls"](env, **final_params)
        return env
    
    def format_actions(self, actions: np.ndarray | T.Tensor):
        """
        Format actions for Isaac Sim environment.
        
        Args:
            actions: Actions to format.
            
        Returns:
            Any: Formatted actions.
        """
        if isinstance(actions, np.ndarray):
            return T.tensor(actions, dtype=T.float32)
        return actions

    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space

    @property
    def single_action_space(self):
        return self.env.single_action_space

    @property
    def single_observation_space(self):
        return self.env.single_observation_space
    
    def reset(self):
        if self.seed is not None:
            return self.env.reset(seed=self.seed)
        return self.env.reset()

    def close(self):
        self.env.close()
        self.app.close()

    def step(self, action: T.Tensor):
        states, rewards, terms, truncs, info = self.env.step(action)
        dones = T.logical_or(terms, truncs)
        return states, rewards, dones, info

    @property
    def config(self):
        return {
            "type": self.__class__.__name__,
            "cfg": self.cfg,
            "num_envs": self.num_envs,
            "wrappers": self.wrappers if self.wrappers else [],
            "render_mode": self.render_mode,
            "seed": self.seed,
            "obs_key": self.obs_key,
            "goal_key": self.goal_key,
        }

    def to_json(self):
        return json.dumps(self.config)

    @classmethod
    def from_json(cls, json_string):
        config = json.loads(json_string)
        cfg = config['cfg']
        # cfg = ManagerBasedRLEnvCfg(**cfg_dict)
        wrappers = config.get("wrappers", [])
        return cls(cfg, wrappers)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, EnvSpec):
            return serialize_env_spec(obj)
        if isinstance(obj, WrapperSpec):
            return wrapper_to_dict(obj)
        if isinstance(obj, GymnasiumWrapper):
            return {
                "type": obj.__class__.__name__,
                "env": obj.env_spec.to_json(),
                "wrappers": obj.wrappers if obj.wrappers else []
            }
        if callable(obj):
            return str(obj)  # Convert functions, including lambdas, to strings

        # Let the base class default method raise the TypeError for unknown types
        return json.JSONEncoder.default(self, obj)

def wrapper_to_dict(wrapper_spec):
    if isinstance(wrapper_spec, WrapperSpec):
        # Convert WrapperSpec to a dictionary dynamically
        wrapper_dict = {}
        for attr in dir(wrapper_spec):
            if not attr.startswith('__') and not callable(getattr(wrapper_spec, attr)):
                wrapper_dict[attr] = getattr(wrapper_spec, attr)
            elif callable(getattr(wrapper_spec, attr)):
                wrapper_dict[attr] = str(getattr(wrapper_spec, attr))  # Convert callable to string
        return wrapper_dict
    return str(wrapper_spec)

def serialize_env_spec(env_spec):
    """Extracts and serializes the relevant parts of the environment specification."""
    env_spec_dict = {
        "id": env_spec.id,
        "entry_point": env_spec.entry_point,
        "reward_threshold": env_spec.reward_threshold,
        "nondeterministic": env_spec.nondeterministic,
        "max_episode_steps": env_spec.max_episode_steps,
        "order_enforce": env_spec.order_enforce,
        "disable_env_checker": env_spec.disable_env_checker,
        "kwargs": env_spec.kwargs,
        "additional_wrappers": [],
        "vector_entry_point": env_spec.vector_entry_point,
    }
    return env_spec_dict

