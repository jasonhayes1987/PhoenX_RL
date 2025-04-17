from typing import Dict, Any
from models import ValueModel, StochasticContinuousPolicy, ActorModel, CriticModel, StochasticDiscretePolicy
from env_wrapper import EnvWrapper
from buffer import Buffer, ReplayBuffer, PrioritizedReplayBuffer
from noise import Noise
from normalizer import Normalizer
from rl_callbacks import load as callback_load
from schedulers import ScheduleWrapper

def load_agent_from_config(config: Dict[str, Any], load_weights: bool = True):
    """
    Load an agent from a configuration dictionary.
    
    Args:
        config: Configuration dictionary
        load_weights: Whether to load the model weights
        
    Returns:
        The loaded agent
    """
    agent_type = config.get("agent_type")
    if agent_type is None:
        raise ValueError("agent_type must be specified in config")
        
    agent_class = get_agent_class_from_type(agent_type)
    if agent_class is None:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    return agent_class.load(config, load_weights)

def get_agent_class_from_type(agent_type: str):
    """
    Get the agent class from its type name.
    
    Args:
        agent_type: The type name of the agent
        
    Returns:
        The agent class
    """
    from rl_agents import PPO, DDPG, Reinforce, ActorCritic, TD3, HER
    agent_classes = {
        "PPO": PPO,
        "DDPG": DDPG,
        "Reinforce": Reinforce,
        "ActorCritic": ActorCritic,
        "TD3": TD3,
        "HER": HER
    }
    return agent_classes.get(agent_type) 