import json
from abc import ABC, abstractmethod
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import EnvSpec, WrapperSpec
from gymnasium.wrappers import (
    AtariPreprocessing,
    TimeLimit,
    TimeAwareObservation,
    FrameStackObservation,
    ResizeObservation
)
from gymnasium.vector import VectorEnv, SyncVectorEnv

class NStepTrajectory(gym.Wrapper):
    def __init__(self, env, n):
        """Initialize the wrapper with the environment and number of steps to track.

        Args:
            env (gym.Env): The Gymnasium environment to wrap.
            n (int): The number of previous steps to include in the trajectory.
        """
        super().__init__(env)
        self.n = n
        self.n_states = deque(maxlen=self.n)
        self.n_actions = deque(maxlen=self.n)
        self.n_rewards = deque(maxlen=self.n)
        self.n_next_states = deque(maxlen=self.n)
        self.n_dones = deque(maxlen=self.n)
        self.current_state = None

    def reset(self, **kwargs):
        """Reset the environment and clear the trajectory history.

        Args:
            **kwargs: Additional arguments for env.reset().

        Returns:
            tuple: (observation, info) from the environment reset.
        """
        # Capture info data to return current trajectories before reset erases info dict
        info = {}
        info['n-step trajectory'] = {
            'states': np.array(self.n_states),
            'actions': np.array(self.n_actions),
            'rewards': np.array(self.n_rewards),
            'next_states': np.array(self.n_next_states),
            'dones': np.array(self.n_dones)
        }
        state, _ = self.env.reset(**kwargs)
        self.n_states = deque(maxlen=self.n)
        self.n_states.extend([np.zeros(state.shape) for _ in range(self.n)])
        self.n_actions = deque(maxlen=self.n)
        self.n_actions.extend([np.zeros(self.env.action_space.shape) for _ in range(self.n)])
        self.n_rewards = deque(maxlen=self.n)
        self.n_rewards.extend([0] * self.n)
        self.n_next_states = deque(maxlen=self.n)
        self.n_next_states.extend([np.zeros(state.shape) for _ in range(self.n)])
        self.n_dones = deque(maxlen=self.n)
        self.n_dones.extend([0] * self.n)
        self.current_state = state

        return state, info

    def step(self, action):
        """Step the environment and update the n-step trajectory.

        Args:
            action: The action to take in the environment.

        Returns:
            tuple: (observation, reward, terminated, truncated, info) with updated info dict.
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        # Append the current step's data to the trajectory
        self.n_states.append(self.current_state)
        self.n_actions.append(action)
        self.n_rewards.append(reward)
        self.n_next_states.append(next_state)
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
        # Add the trajectory to the info dictionary
        info['n-step trajectory'] = trajectory

        return next_state, reward, terminated, truncated, info

WRAPPER_REGISTRY = {
    "AtariPreprocessing": {
        "cls": AtariPreprocessing,
        "default_params": {
            "frame_skip": 1,
            "grayscale_obs": True,
            "scale_obs": True
        }
    },
    "TimeLimit": {
        "cls": TimeLimit,
        "default_params": {
            "max_episode_steps": 1000
        }
    },
    "TimeAwareObservation": {
        "cls": TimeAwareObservation,
        "default_params": {
            "flatten": False,
            "normalize_time": False
        }
    },
    "FrameStackObservation": {
        "cls": FrameStackObservation,
        "default_params": {
            "stack_size": 4
        }
    },
    "ResizeObservation": {
        "cls": ResizeObservation,
        "default_params": {
            "shape": 84
        }
    },
    "NStepTrajectory": {
        "cls": NStepTrajectory,
        "default_params": {"n": 1}
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
    def render(self, mode="rgb_array"):
        """
        Render the environment.

        Args:
            mode (str): The render mode (default: "rgb_array").

        Returns:
            Any: Rendered frame or visualization.
        """
        pass

    @abstractmethod
    def _initialize_env(self, render_freq: int = 0, num_envs: int = 1, seed: int = None):
        """
        Initialize the environment with optional rendering and seeding.

        Args:
            render_freq (int): Frequency of rendering (default: 0).
            num_envs (int): Number of parallel environments (default: 1).
            seed (int): Random seed for the environment (default: None).

        Returns:
            Any: The initialized environment.
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
    def __init__(self, env_spec: EnvSpec, wrappers: list[dict] = None, worker_id: int = 0):
        self.env_spec = env_spec
        self.wrappers = wrappers
        self.worker_id = worker_id
        self.traj_counters = []  # Per-environment counters
        self.unique_env_ids = []  # Unique IDs for each env
        self.num_envs = 1
        self.traj_ids = []
        self.step_indices = []
        self.env = self._initialize_env()
        

    def _initialize_env(self, render_freq: int = 0, num_envs: int = 1, seed: int = None):
        """
        Initialize the Gymnasium environment with unique seeds for each environment.

        Args:
            render_freq (int): Frequency of rendering (default: 0).
            num_envs (int): Number of parallel environments (default: 1).
            seed (int): Base random seed for the environment (default: None).

        Returns:
            gym.Env: The initialized Gymnasium environment.
        """
        self.seed = seed
        if self.seed is None:
            seeds = [None] * num_envs
        else:
            seeds = [self.seed + i for i in range(num_envs)]  # Create different seeds for each environment
        
        # Create a list of environment factories, each with its unique seed
        env_fns = []
        for i in range(num_envs):
            def make_env(i=i):  # Use default argument to capture i
                env = gym.make(self.env_spec.id, render_mode="rgb_array" if render_freq > 0 else None)
                if seeds[i] is not None:
                    env.reset(seed=seeds[i])  # Set seed for each environment
                    env.action_space.seed(seeds[i])  # Also seed the action space
                if self.wrappers:
                    for wrapper in self.wrappers:
                        if wrapper['type'] in WRAPPER_REGISTRY:
                            default_params = WRAPPER_REGISTRY[wrapper['type']]["default_params"].copy()
                            override_params = wrapper.get("params", {})
                            final_params = {**default_params, **override_params}
                            env = WRAPPER_REGISTRY[wrapper['type']]["cls"](env, **final_params)
                return env
            
            env_fns.append(make_env)

        vec_env = SyncVectorEnv(env_fns)

        # Initialize self.num_envs and internal env tracking
        self.num_envs = num_envs
        self.traj_counters = [0] * num_envs
        self.unique_env_ids = [(self.worker_id * num_envs) + i for i in range(num_envs)]
        self.traj_ids = [self._compute_traj_id(i) for i in range(num_envs)]
        self.step_indices = [0] * num_envs

        return vec_env
    
    def _compute_traj_id(self, env_idx):
        return (self.unique_env_ids[env_idx] << 32) + self.traj_counters[env_idx]

    def reset(self):
        if self.seed is not None:
            return self.env.reset(seed=self.seed)
        return self.env.reset()

    def step(self, action, testing=False):
        states, rewards, terms, truncs, infos = self.env.step(action)
        dones = np.logical_or(terms, truncs)
        return states, rewards, dones, infos
    
        # if testing:
        #     return states, rewards, dones, infos
        # else:
        #     for i in range(self.num_envs):
        #         if dones[i]:
        #             self.traj_counters[i] += 1
        #             self.traj_ids[i] = self._compute_traj_id(i)
        #             self.step_indices[i] = 0
        #         else:
        #             self.step_indices[i] += 1
        #     return states, rewards, dones, infos, self.traj_ids, self.step_indices
    
    def render(self, mode="rgb_array"):
        """
        Render the environment.

        Args:
            mode (str): The render mode (default: "rgb_array").

        Returns:
            Any: Rendered frame or visualization.
        """
        return self.env.render(mode=mode)
    
    def format_actions(self, actions: np.ndarray, testing=False):
        if isinstance(self.action_space, gym.spaces.Box):
            if testing:
                num_envs = 1
            else:
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
            "wrappers": self.wrappers,
            "worker_id": self.worker_id
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
        #DEBUG
        # print('GymnasiumWrapper from_json called')
        # print(f'from json env spec:{json_env_spec}, type:{type(json_env_spec)}')
        config = json.loads(json_env_spec)
        #DEBUG
        # print(f'from json config:{config}, type:{type(config)}')
        env_spec = EnvSpec.from_json(config['env'])
        #DEBUG
        # print(f'wrappers in gym from json:{config["wrappers"]}')
        try:
            return cls(env_spec, config["wrappers"], config["worker_id"])
        except Exception as e:
            raise ValueError(f"Environment wrapper error: {config}, {e}")
    
class IsaacSimWrapper(EnvWrapper):
    def __init__(self, env_spec):
        """
        Placeholder wrapper for Isaac Sim environments.

        This class is a template and needs implementation based on Isaac Sim's API.
        """
        pass

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

