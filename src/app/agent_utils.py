from typing import Dict, Any
from models import ValueModel, StochasticContinuousPolicy, ActorModel, CriticModel, StochasticDiscretePolicy
from env_wrapper import EnvWrapper
from buffer import Buffer, ReplayBuffer, PrioritizedReplayBuffer
from noise import Noise
from normalizer import Normalizer
from rl_callbacks import load as callback_load, WandbCallback, RayWandbCallback
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
        
    agent = agent_class.load(config, load_weights)
        
    return agent

def convert_to_distributed_callbacks(callbacks, role: str, worker_id=0):
    """
    Convert standard callbacks to distributed-friendly versions
    
    Args:
        callbacks (list): List of callback objects
        worker_id (int): Worker ID for this process
        
    Returns:
        list: Modified callbacks for distributed training
    """
    if not callbacks:
        return callbacks
        
    distributed_callbacks = []
    
    for callback in callbacks:
        if isinstance(callback, WandbCallback):
            config = callback.get_config()
            # Replace with RayWandbCallback
            ray_wandb_callback = RayWandbCallback(
                project_name=config["config"]["project_name"],
                role=role,
                run_name=config["config"]["run_name"],
                chkpt_freq=config["config"]["chkpt_freq"],
                worker_id=worker_id,
                _sweep=config["config"]["_sweep"]
            )
            
            
            distributed_callbacks.append(ray_wandb_callback)
        else:
            # Keep other callbacks as-is
            distributed_callbacks.append(callback)
            
    return distributed_callbacks

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