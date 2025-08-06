import json
import gymnasium as gym
from gymnasium.envs.registration import EnvSpec, WrapperSpec
from env_wrapper import GymnasiumWrapper

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        #DEBUG
        print(f'custom encoder obj: {obj}')
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
        # "additional_wrappers": [wrapper_to_dict(wrapper) for wrapper in env_spec.additional_wrappers],
        "additional_wrappers": [],
        "vector_entry_point": env_spec.vector_entry_point,
    }
    return env_spec_dict
