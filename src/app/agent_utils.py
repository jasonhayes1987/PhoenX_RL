from typing import Dict, Any, List, Optional
import torch as T
import torch.nn as nn
from models import ValueModel, StochasticContinuousPolicy, ActorModel, CriticModel, StochasticDiscretePolicy
from env_wrapper import EnvWrapper
from buffer import Buffer, ReplayBuffer, PrioritizedReplayBuffer
from noise import Noise
from normalizer import Normalizer
from rl_callbacks import load as callback_load, WandbCallback, RayWandbCallback
from schedulers import ScheduleWrapper

def compute_n_step_return(
    rewards: T.Tensor,           # [batch_size, N]
    dones: T.Tensor,            # [batch_size, N]
    gamma: float,
    N: int,
    device: str = "cpu"
) -> T.Tensor:
    """
    Compute N-step returns for a batch of sequences.

    Args:
        rewards: Tensor of rewards [batch_size, N].
        dones: Tensor of done flags [batch_size, N].
        gamma: Discount factor.
        N: Number of steps for the return.
        device: Device for tensor operations.

    Returns:
        Tensor of N-step returns [batch_size].
    """
    batch_size = rewards.size(0)
    #DEBUG
    # print(f'rewards:{rewards}, dones:{dones}, gamma:{gamma}, N:{N}, device:{device}')

    # Discount factors: [1, gamma, gamma^2, ..., gamma^{N-1}]
    discount_factors = T.pow(gamma, T.arange(N, device=device).float()).unsqueeze(0).expand(batch_size, N)
    #DEBUG
    # print(f'discount_factors:{discount_factors}')

    # Cumulative done mask: 1 if any 'done' up to step k
    # cum_done = T.cumsum(dones, dim=1).float()
    # Include rewards[k] if no 'done' up to k-1 (always include k=0)
    # include_mask = (cum_done == 0).float()
    # include_mask[:, 0] = 1.0

    # Compute masked discounted rewards
    masked_rewards = rewards * discount_factors # * include_mask
    #DEBUG
    # print(f'masked_rewards:{masked_rewards}')
    return_ = masked_rewards.sum(dim=1)
    #DEBUG
    # print(f'return_:{return_}')

    return return_

# def compute_n_step_return(
#     rewards: T.Tensor,           # [batch_size, N]
#     dones: T.Tensor,            # [batch_size, N]
#     gamma: float,
#     N: int,
#     device: str = "cpu"
# ) -> T.Tensor:
#     """
#     Compute N-step returns for a batch of sequences.

#     Args:
#         rewards: Tensor of rewards [batch_size, N].
#         dones: Tensor of done flags [batch_size, N].
#         gamma: Discount factor.
#         N: Number of steps for the return.
#         device: Device for tensor operations.

#     Returns:
#         Tensor of N-step returns [batch_size].
#     """
#     batch_size = rewards.size(0)

#     # Discount factors: [1, gamma, gamma^2, ..., gamma^{N-1}]
#     discount_factors = T.pow(gamma, T.arange(N, device=device).float()).unsqueeze(0).expand(batch_size, N)

#     # Compute cumulative done up to step k-1
#     # if N > 1:
#     #     cum_done_up_to_k_minus_1 = T.cat(
#     #         [T.zeros(batch_size, 1, device=device), T.cumsum(dones[:, :-1], dim=1)], dim=1
#     #     )
#     # else:
#     #     cum_done_up_to_k_minus_1 = T.zeros(batch_size, 1, device=device)
#     # include_mask = (cum_done_up_to_k_minus_1 == 0).float()

#     # Compute masked discounted rewards
#     # masked_rewards = rewards * discount_factors * include_mask
#     masked_rewards = rewards * discount_factors * (1 - dones)
#     return_ = masked_rewards.sum(dim=1)

#     return return_

def compute_full_return(rewards, gamma):
    """
    Compute discounted returns for each step in a trajectory.
    
    Args:
        rewards (list[float]): List of rewards from the trajectory.
        gamma (float): Discount factor (0 <= gamma <= 1).
    
    Returns:
        list[float]: List of discounted returns for each step.
    """
    returns = []
    discounted_return = 0.0
    for reward in reversed(rewards):
        discounted_return = reward + gamma * discounted_return
        returns.append(discounted_return)
    returns.reverse()
    return returns

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
        role (str): 'learner' or 'worker'
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
