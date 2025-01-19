import json
from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

def atari_wrappers(env):
    """
    Wrap an Atari environment with preprocessing and frame stacking.

    This function applies standard Atari preprocessing, including converting to grayscale,
    resizing, scaling, and stacking multiple consecutive frames for better temporal
    context.

    Args:
        env (gym.Env): The original Atari environment.

    Returns:
        gym.Env: The wrapped environment with preprocessing and frame stacking applied.
    """
    env = AtariPreprocessing(
        env,
        frame_skip=1,
        grayscale_obs=True,
        scale_obs=True,
        screen_size=84
    )
    env = FrameStackObservation(env, stack_size=4)
    return env

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
    def __init__(self, env_spec: EnvSpec, wrappers: list[callable] = None):
        self.env_spec = env_spec
        self._wrappers = wrappers or []
        self.env = self._initialize_env()

    def _initialize_env(self, render_freq: int = 0, num_envs: int = 1, seed: int = None):
        """
        Initialize the Gymnasium environment.

        Args:
            render_freq (int): Frequency of rendering (default: 0).
            num_envs (int): Number of parallel environments (default: 1).
            seed (int): Random seed for the environment (default: None).

        Returns:
            gym.Env: The initialized Gymnasium environment.
        """
        self.seed = seed
        env = gym.make_vec(
            id=self.env_spec,  # Can use EnvSpec directly here
            num_envs=num_envs,
            wrappers=self._wrappers,
            render_mode="rgb_array" if render_freq > 0 else None,
        )

        if self.seed is not None:
            _,_ = env.reset(seed=self.seed)
        
        return env

    def reset(self):
        """
        Reset the environment.

        Returns:
            Any: Initial observation of the environment.
        """
        if self.seed is not None:
            return self.env.reset(seed=self.seed)
        return self.env.reset()
    
    def step(self, action):
        """
        Take an action in the environment.

        Args:
            action: The action to be taken.

        Returns:
            Tuple: Observation, reward, done flag, and additional info.
        """
        return self.env.step(action)
    
    def render(self, mode="rgb_array"):
        """
        Render the environment.

        Args:
            mode (str): The render mode (default: "rgb_array").

        Returns:
            Any: Rendered frame or visualization.
        """
        return self.env.render(mode=mode)
    
    def format_actions(self, actions: np.ndarray):
        if isinstance(self.action_space, gym.spaces.Box):
            num_envs = self.env.num_envs
            num_actions = self.action_space.shape[-1]
            return actions.reshape(num_envs, num_actions)
        if isinstance(self.action_space, gym.spaces.Discrete) or isinstance(self.action_space, gym.spaces.MultiDiscrete):
            return actions.ravel()
    
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
            "env": self.env_spec.to_json()
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
        try:
            return cls(env_spec)
        except Exception as e:
            raise ValueError(f"Environment wrapper error: {config}, {e}")
    
class IsaacSimWrapper(EnvWrapper):
    def __init__(self, env_spec):
        """
        Placeholder wrapper for Isaac Sim environments.

        This class is a template and needs implementation based on Isaac Sim's API.
        """
        pass