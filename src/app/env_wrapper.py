import json
from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium.envs.registration import EnvSpec

class EnvWrapper(ABC):

    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def step(self, action):
        pass
    
    @abstractmethod
    def render(self, mode="rgb_array"):
        pass

    @abstractmethod
    def _initialize_env(self, env_spec, render_freq: int = 0, num_envs: int = 1, seed: int = None):
        pass
    
    @property
    @abstractmethod
    def observation_space(self):
        pass
    
    @property
    @abstractmethod
    def action_space(self):
        pass

    @abstractmethod
    def to_json(self) -> str:
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, json_string: str):
        pass


class GymnasiumWrapper(EnvWrapper):
    def __init__(self, env_spec:dict):
        self.env = self._initialize_env(env_spec)

    # def _initialize_env(self, env_spec: dict, render_freq: int = 0, num_envs: int = 1, seed: int = None):
    #     env_spec_obj = EnvSpec.from_json(env_spec)
    #     #DEBUG
    #     print(f'_initialize_env env_spec:{env_spec}')
    #     print(f'_initialize_env env_obj:{env_spec_obj}')
    #     if render_freq > 0:
    #         env = gym.vector.SyncVectorEnv([
    #             lambda: gym.make(env_spec_obj, render_mode="rgb_array")
    #             for _ in range(num_envs)
    #         ])
    #     else:
    #         env = gym.vector.SyncVectorEnv([
    #             lambda: gym.make(env_spec_obj)
    #             for _ in range(num_envs)
    #         ])
        
    #     if seed is not None:
    #         gym.utils.seeding.np_random.seed = seed
    #         _, _ = env.reset(seed=seed)
    #         _ = env.action_space.seed(seed)
        
    #     return env

    def _initialize_env(self, env_spec: dict, render_freq: int = 0, num_envs: int = 1, seed: int = None):
        #DEBUG
        # print(f'gymnasium _initialize_env called with:')
        env_spec_obj = EnvSpec.from_json(env_spec)
        #DEBUG
        # print(f'_initialize_env env_spec:{env_spec}')
        # print(f'render_freq:{render_freq}')
        # print(f'seed:{seed}')
        # print(f'num envs:{num_envs}')

        # print(f'_initialize_env env_obj:{env_spec_obj}')
        
        # vectorization_mode = "sync" if render_freq == 0 else "async"  # Example choice, adjust as needed
        # vector_kwargs = {}  # Any additional kwargs for vector environment
        
        env = gym.make_vec(
            id=env_spec_obj,  # Can use EnvSpec directly here
            num_envs=num_envs,
            # vectorization_mode=vectorization_mode,
            render_mode="rgb_array" if render_freq > 0 else None,
            # **vector_kwargs
        )

        if seed is not None:
            gym.utils.seeding.np_random.seed = seed
            _, _ = env.reset(seed=seed)
            _ = env.action_space.seed(seed)

        #DEBUG
        # print(f'env created:{env}')
        
        return env

    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self, mode="rgb_array"):
        return self.env.render(mode=mode)
    
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
    
    @property
    def config(self):
        return {
            "type": self.__class__.__name__,
            "env": self.env.envs[0].spec.to_json()
        }
    
    def to_json(self):
        return json.dumps(self.config)

    @classmethod
    def from_json(cls, json_env_spec):
        config = json.loads(json_env_spec)
        try:
            return cls(config)
        except Exception as e:
            raise ValueError(f"Environment wrapper error: {config}, {e}")
    

# Note: This is a placeholder. You would need to implement this based on Isaac Sim's API.
class IsaacSimWrapper(EnvWrapper):
    def __init__(self, env_spec):
        # Setup Isaac Sim environment here based on env_spec
        pass
    # Implement abstract methods similarly to GymnasiumWrapper