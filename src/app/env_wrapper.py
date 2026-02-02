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
from gymnasium.vector import VectorEnv, SyncVectorEnv, VectorWrapper, utils

from app.torch_utils import get_device

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

class VectorNStepReward(VectorWrapper):
    def __init__(self, env, n: int, obs_key: str | None = None, goal_key: str | None = None, ach_goal_key: str | None = None):
        """
        Initialize the vectorized wrapper for n-step trajectories.
        Args:
            env (gym.VectorEnv | ManagerBasedRLEnv): The vectorized environment to wrap.
            n (int): The number of previous steps to include in the trajectory.
            obs_key (str): The key for the observation space.
            goal_key (str | None): The key for the goal space.
            ach_goal_key (str | None): The key for the achieved goal space.
        """
        self.env = env
        self.n = n
        self.obs_key = obs_key
        self.goal_key = goal_key
        self.ach_goal_key = ach_goal_key

        if isinstance(env, gym.vector.VectorEnv):
            super().__init__(env)
        else:
            self.observation_space = utils.batch_space(self.single_observation_space, self.num_envs)
            self.action_space = utils.batch_space(self.single_action_space, self.num_envs)

        # Per env deques for trajectories
        self.n_states = [deque(maxlen=self.n) for _ in range(self.num_envs)]
        self.n_actions = [deque(maxlen=self.n) for _ in range(self.num_envs)]
        self.n_rewards = [deque(maxlen=self.n) for _ in range(self.num_envs)]
        self.n_next_states = [deque(maxlen=self.n) for _ in range(self.num_envs)]
        self.n_dones = [deque(maxlen=self.n) for _ in range(self.num_envs)]

        if self.goal_key:
            self.n_state_achieved_goals = [deque(maxlen=self.n) for _ in range(self.num_envs)]
            self.n_next_state_achieved_goals = [deque(maxlen=self.n) for _ in range(self.num_envs)]
            self.n_desired_goals = [deque(maxlen=self.n) for _ in range(self.num_envs)]

        self.current_states = None

        # Set internal attributes
        device = get_device()
        state_shape = self.single_observation_space[self.obs_key].shape if self.obs_key is not None else self.single_observation_space.shape
        self._pad_state = T.zeros(state_shape, dtype=T.float32, device=device)
        self._pad_action = T.zeros(self.single_action_space.shape, dtype=T.float32, device=device)
        self._pad_reward = T.tensor(0.0, dtype=T.float32, device=device)
        self._pad_done = T.tensor(0.0, dtype=T.float32, device=device)
        if self.goal_key:
            self._pad_goal = T.zeros(self.single_observation_space[self.goal_key].shape, dtype=T.float32, device=device)

    def reset(self, **kwargs):
        """
        Reset all envs and clear per-env trajectories.
        Returns batched (observations, infos)
        """

        states, infos = self.env.reset(**kwargs)

        # Capture existing trajectories
        # trajectory = self.build_trajectories()

        # Clear env deques
        for i in range(self.num_envs):
            self.n_states[i].clear()
            self.n_actions[i].clear()
            self.n_rewards[i].clear()
            self.n_next_states[i].clear()
            self.n_dones[i].clear()
            if self.goal_key:
                self.n_state_achieved_goals[i].clear()
                self.n_next_state_achieved_goals[i].clear()
                self.n_desired_goals[i].clear()

        self.current_states = states

        if 'n-step trajectory' not in infos:
            infos['n-step trajectory'] = {}
        # infos['n-step trajectory'].update(trajectory)

        return states, infos
    
    def step(self, actions: T.Tensor):
        """
        Step all envs with batched actions, update per-env trajectories, clears env trajectory deques if done is True
        Returns batched (next_states, rewards, terminations, truncations, infos)
        """
        device = get_device()
        next_states, rewards, terminations, truncations, infos = self.env.step(actions)
        dones = terminations | truncations
        
        # ensure tensors
        rewards, dones, actions = (
            T.as_tensor(rewards, device=device) if isinstance(rewards, np.ndarray) else rewards,
            T.as_tensor(dones, device=device) if isinstance(dones, np.ndarray) else dones,
            T.as_tensor(actions, device=device) if isinstance(actions, np.ndarray) else actions,
        )

        for i in range(self.num_envs):
            # extract states
            if dones[i].item() and infos.get("_final_obs", [False]*self.num_envs)[i]:
                next_state = infos['final_obs'][i][self.obs_key] if self.obs_key is not None else infos['final_obs'][i]
                if self.goal_key:
                    goal = self.current_states[self.goal_key][i]
                    ach_goal = self.current_states[self.ach_goal_key][i]
                    next_ach_goal = infos['final_obs'][i][self.ach_goal_key]
            else:
                next_state = next_states[self.obs_key][i] if self.obs_key is not None else next_states[i]
                if self.goal_key:
                    goal = self.current_states[self.goal_key][i]
                    ach_goal = self.current_states[self.ach_goal_key][i]
                    next_ach_goal = next_states[self.ach_goal_key][i]
            state = self.current_states[self.obs_key][i] if self.obs_key is not None else self.current_states[i]
            
            # ensure tensors
            state, next_state = (
                T.as_tensor(state, device=device) if isinstance(state, np.ndarray) else state,
                T.as_tensor(next_state, device=device) if isinstance(next_state, np.ndarray) else next_state,
            )

            if self.goal_key:
                goal, ach_goal, next_ach_goal = (
                    T.as_tensor(goal, device=device) if isinstance(goal, np.ndarray) else goal,
                    T.as_tensor(ach_goal, device=device) if isinstance(ach_goal, np.ndarray) else ach_goal,
                    T.as_tensor(next_ach_goal, device=device) if isinstance(next_ach_goal, np.ndarray) else next_ach_goal,
                )

            # Append current step
            self.n_states[i].append(state)
            self.n_actions[i].append(actions[i])
            self.n_rewards[i].append(rewards[i])
            self.n_next_states[i].append(next_state)
            self.n_dones[i].append(dones[i])
            if self.goal_key:
                self.n_state_achieved_goals[i].append(ach_goal if self.ach_goal_key is not None else None)
                self.n_next_state_achieved_goals[i].append(next_ach_goal if self.ach_goal_key is not None else None)
                self.n_desired_goals[i].append(goal if self.goal_key is not None else None)

        # Build batched trajectory
        trajectory = self.build_trajectories()
        infos['n-step trajectory'] = trajectory
        
        # Clear done trajectories
        for i in range(self.num_envs):
            if dones[i].item():
                self.n_states[i].clear()
                self.n_actions[i].clear()
                self.n_rewards[i].clear()
                self.n_next_states[i].clear()
                self.n_dones[i].clear()
                if self.goal_key:
                    self.n_state_achieved_goals[i].clear()
                    self.n_next_state_achieved_goals[i].clear()
                    self.n_desired_goals[i].clear()

        self.current_states = next_states

        return next_states, rewards, terminations, truncations, infos

    def build_trajectories(self):
        """Construct batched n-step trajectory dict from per-env deques."""
        device = get_device()

        states = self.format_trajectory(self.n_states, pad_mode="repeat")
        next_states = self.format_trajectory(self.n_next_states, pad_mode="repeat")
        actions = self.format_trajectory(self.n_actions, pad_mode="repeat")
        rewards = self.format_trajectory(self.n_rewards, pad_mode=T.tensor(0.0, dtype=T.float32, device=device))
        dones = self.format_trajectory(self.n_dones, pad_mode=T.tensor(0.0, dtype=T.float32, device=device))
        if self.goal_key:
            desired_goals = self.format_trajectory(self.n_desired_goals, pad_mode="repeat")
            state_achieved_goals = self.format_trajectory(self.n_state_achieved_goals, pad_mode="repeat")
            next_state_achieved_goals = self.format_trajectory(self.n_next_state_achieved_goals, pad_mode="repeat")

        # Determine actual trajectory lengths to compute n-step returns
        lengths = []
        for d in self.n_dones:
            lengths.append(len(d))
        lengths = T.tensor(lengths, device=device)
        
        trajectory = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'trajectory_lengths': lengths,
        }
        if self.goal_key:
            trajectory['state_achieved_goals'] = state_achieved_goals
            trajectory['next_state_achieved_goals'] = next_state_achieved_goals
            trajectory['desired_goals'] = desired_goals
        return trajectory

    def format_trajectory(self, trajectory: List[deque[T.Tensor]], pad_mode:str|T.Tensor="repeat"):
        """Format trajectory from per-env deques to batched tensor.
        
        Args:
            trajectory: List of deques containing tensors.
            pad_mode: Mode to pad the trajectory. "repeat" to repeat the last value or tensor to pad with passed value.

        Returns:
            Tensor: Batched trajectory.
        """
        trajs = []
        for d in trajectory:
            seq = list(d)
            if pad_mode == "repeat":
                padding = seq[-1]
            else:
                padding = pad_mode
            while len(seq) < self.n:
                seq.append(padding)
            trajs.append(T.stack(seq, dim=0))
        return T.stack(trajs, dim=0)

    @property
    def single_action_space(self):
        return self.env.single_action_space

    @property
    def single_observation_space(self):
        return self.env.single_observation_space

class OneHotObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.observation_space, gym.spaces.Discrete), "Observation space must be Discrete."
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.observation_space.n,), dtype=np.float32)
    
    def observation(self, obs):
        one_hot = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        one_hot[obs] = 1.0
        return one_hot


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
        "vector_aware": True,
        "default_params": {"n": 1, "obs_key": None, "goal_key": None, "ach_goal_key": None}
    },
    "OneHotObservationWrapper": {
        "cls": OneHotObservationWrapper,
        "default_params": {}
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

    def clone(self, num_envs:int=1, **kwargs) -> 'EnvWrapper':
        """
        Create a new instance of the environment wrapper with the passed parameters.

        Args:
            **kwargs: Additional keyword arguments to pass to the environment wrapper to override original values.

        Returns:
            EnvWrapper: A new instance of the environment wrapper with the passed parameters.
        """
        config = json.loads(self.to_json())
        config['config'].update(num_envs=num_envs, **kwargs)
        return self.from_json(json.dumps(config))

    @abstractmethod
    def format_actions(self, actions: np.ndarray | T.Tensor):
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
            if config['type'] == 'gymnasium':
                return GymnasiumWrapper.from_json(json_string)
            # Add more conditions here for other subclasses if they exist
            elif config['type'] == 'isaacsim':
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
    def __init__(self, cfg:str, num_envs:int=1, wrappers:list[dict]|None=None,
                 render_mode:str|None=None, seed:int|None=None, obs_key:str|None=None, goal_key:str|None=None):
        self.env_id = cfg
        self.wrappers = wrappers
        self.num_envs = num_envs
        if seed is None:
            seed = np.random.randint(1000)
        self.seed = seed
        self.render_mode = render_mode
        self.obs_key = obs_key
        self.goal_key = goal_key
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
                # final_params.update({"obs_key": self.obs_key, "goal_key": self.goal_key})
                
                if vector_aware:
                    vector_wrappers.append((cls, final_params))
                else:
                    def wrapper_fn(env, cls=cls, params=final_params):
                        return cls(env, **params)
                    single_wrappers.append(wrapper_fn)

        # Create vector env with single-env wrappers applied per sub-env
        vec_env = gym.make_vec(
            id=self.env_id,
            num_envs=self.num_envs,
            vectorization_mode="sync",
            vector_kwargs={"autoreset_mode": "SameStep"},
            wrappers=single_wrappers,
            render_mode=self.render_mode
        )

        # Apply vector-aware wrappers to the entire vec_env
        for cls, params in vector_wrappers:
            vec_env = cls(vec_env, **params)

        return vec_env

    def render_frame(self)->np.ndarray:
        """Renders a frame from the environment.
        
        Returns:
            np.ndarray: The rendered frame.
        """
        frame = self.env.render()        
        return frame[0]
        

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
        states, rewards, terminations, truncations, infos = self.env.step(action)
        dones = terminations | truncations
        
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
            "type": "gymnasium",
            "config":{
                "cfg": self.env_id,
                "num_envs": self.num_envs,
                "wrappers": self.wrappers,
                "render_mode": self.render_mode,
                "seed": self.seed,
                "obs_key": self.obs_key,
                "goal_key": self.goal_key,
            }
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
        config = config['config']
        #DEBUG
        print(f'GymnasiumWrapper from_json config:{config}')
        try:
            return cls(**config)
        except Exception as e:
            raise ValueError(f"Environment wrapper error: {config}, {e}")
    
class IsaacSimWrapper(EnvWrapper):
    def __init__(self, cfg:str, num_envs:int=1, wrappers:list[dict]|None=None,
                 render_mode:str='headless', seed:int|None=None, obs_key:str='policy', goal_key:str|None=None):
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
        # Initialize env
        self.env = self._initialize_env()
        

    def _initialize_env(self):
        """
        Initialize the Isaac Sim environment with unique seeds for each environment.
        """
        import importlib

        try:
            import omni.kit.app as kit_app  # type: ignore[reportMissingImports]
            self.app = kit_app.get_app()
        except Exception:
            self.app = None
        if self.app is None:
            try:
                from isaaclab.app import AppLauncher  # type: ignore[reportMissingImports]
            except (ModuleNotFoundError, ImportError):
                try:
                    from omni.isaac.lab.app import (  # type: ignore[reportMissingImports]
                        AppLauncher,
                    )
                except (ModuleNotFoundError, ImportError) as e:
                    raise ModuleNotFoundError(
                        "Isaac Lab is required for Isaac Sim environments but could not be imported. "
                        "Install Isaac Lab / Isaac Sim Python packages, or ensure ISAACLAB_PATH is set "
                        "to IsaacLab's `source/` directory so `isaaclab` (or `omni.isaac.lab`) is on PYTHONPATH."
                    ) from e
            app_launcher = AppLauncher(headless=(self.render_mode=='headless'), device="cuda:0", enable_cameras=True)
            self.app = app_launcher.app
        
        try:
            from isaaclab.envs import ManagerBasedRLEnv  # type: ignore[reportMissingImports]
        except (ModuleNotFoundError, ImportError):
            try:
                from omni.isaac.lab.envs import (  # type: ignore[reportMissingImports]
                    ManagerBasedRLEnv,
                )
            except (ModuleNotFoundError, ImportError) as e:
                raise ModuleNotFoundError(
                    "Isaac Lab is required for Isaac Sim environments but could not be imported. "
                    "Expected `isaaclab.envs` or `omni.isaac.lab.envs` to be available."
                ) from e

        module_path, class_name = self.cfg.split(':')
        cfg_class = getattr(importlib.import_module(module_path), class_name)
        cfg = cfg_class()
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
                    final_params.update({"obs_key": self.obs_key, "goal_key": self.goal_key})
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
        states, rewards, terminations, truncations, info = self.env.step(action)
        dones = terminations | truncations

        return states, rewards, dones, info

    @property
    def config(self):
        return {
            "type": "isaacsim",
            "config":{
                "cfg": self.cfg,
                "num_envs": self.num_envs,
                "wrappers": self.wrappers if self.wrappers else [],
                "render_mode": self.render_mode,
                "seed": self.seed,
                "obs_key": self.obs_key,
                "goal_key": self.goal_key,
            }
        }

    def to_json(self):
        return json.dumps(self.config)

    @classmethod
    def from_json(cls, json_string):
        config = json.loads(json_string)
        #DEBUG
        print(f'IsaacSimWrapper from_json config: {config}')
        config = config['config']
        try:
            return cls(**config)
        except Exception as e:
            raise ValueError(f"Environment wrapper error: {config}, {e}")


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, EnvSpec):
            return serialize_env_spec(obj)
        if isinstance(obj, WrapperSpec):
            return wrapper_to_dict(obj)
        if isinstance(obj, GymnasiumWrapper):
            return {
                "type": "gymnasium",
                "env": obj.env_id.to_json(),
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

