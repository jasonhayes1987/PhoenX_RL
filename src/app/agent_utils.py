import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch as T
import torch.nn as nn
from torch.optim import Optimizer
from .models import ActorModel, ContinuousCritic, ValueModel, StochasticContinuousPolicy, StochasticDiscretePolicy, select_policy_model, select_critic_model
from .env_wrapper import EnvWrapper, GymnasiumWrapper, IsaacSimWrapper, NStepReward, VectorNStepReward
from .buffer import Buffer, ReplayBuffer, PrioritizedReplayBuffer
from .noise import Noise, NormalNoise, UniformNoise, OUNoise
from .normalizer import BaseNormalizer, RunningNorm, BatchNorm, RewardNorm
from .rl_callbacks import load as callback_load, WandbCallback, RayWandbCallback
from .schedulers import ScheduleWrapper
from .torch_utils import get_device

def compute_n_step_return(
    rewards: T.Tensor,           # [batch_size, N]
    gamma: float,
    device:
    T.device|str|None = None
) -> T.Tensor:
    """
    Compute N-step returns for a batch of sequences.

    Args:
        rewards: Tensor of rewards [batch_size, N].
        gamma: Discount factor.
        N: Number of steps for the return.
        device: Device for tensor operations.

    Returns:
        Tensor of N-step returns [batch_size].
    """
    device = get_device(device)
    batch_size, N = rewards.shape
    discount_factors = T.pow(gamma, T.arange(N, device=device).float()).unsqueeze(0).expand(batch_size, N)

    return (rewards * discount_factors).sum(dim=1)

def compute_td_error(
    rewards: T.Tensor,
    values: T.Tensor,
    next_values: T.Tensor,
    terminations: T.Tensor,
    truncations: T.Tensor,
    gamma: float,
    bootstrap_truncations: bool
    ) -> T.Tensor:
    """
    Compute TD errors for a batch of trajectories.
    
    Args:
        rewards: Tensor of rewards [batch_size, num_envs].
        values: Tensor of values [batch_size, num_envs].
        next_values: Tensor of next values [batch_size, num_envs].
        terminations: Tensor of termination flags [batch_size, num_envs].
        truncations: Tensor of truncation flags [batch_size, num_envs].
        gamma: Discount factor.
        bootstrap_truncations: Whether to bootstrap the returns on truncated episodes.

    Returns:
        Tensor of TD errors [batch_size, num_envs].
    """
    if bootstrap_truncations:
        dones = terminations
    else:
        dones = T.logical_or(terminations, truncations)
    return rewards + gamma * next_values * T.logical_not(dones) - values

def compute_monte_carlo_returns(
    rewards:T.Tensor,
    gamma:float,
    device:T.device|str|None = None
) -> T.Tensor:
    """
    Compute discounted returns for each step in a trajectory.
    
    Args:
        rewards: Tensor of rewards [batch_size, num_envs].
        gamma: Discount factor.
        device: Device for tensor operations.
    
    Returns:
        Tensor of discounted returns [batch_size, num_envs].
    """
    device = get_device(device)
    returns = []
    discounted_return = 0.0
    for reward in reversed(rewards):
        discounted_return = reward + gamma * discounted_return
        returns.append(discounted_return)
    returns.reverse()
    return T.tensor(returns, device=device)

def compute_gae(
    td_errors:T.Tensor,
    terminations:T.Tensor,
    truncations:T.Tensor,
    gamma:float,
    gae_lambda:float,
    bootstrap_truncations: bool,
    device:T.device|str|None = None
    ) -> T.Tensor:
    """
    Compute Generalized Advantage Estimation (GAE) for a batch of TD errors.
    
    Args:
        td_errors: Tensor of TD errors [timesteps, num_envs].
        terminations: Tensor of termination flags [timesteps, num_envs].
        truncations: Tensor of truncation flags [timesteps, num_envs].
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.
        bootstrap_truncations: Whether to bootstrap the returns on truncated episodes.
        device: Device for tensor operations.
    """
    device = get_device(device)
    timesteps, num_envs = td_errors.shape
    advantages = T.zeros(timesteps, num_envs, device=device)
    advantage = T.zeros(num_envs, device=device)

    if bootstrap_truncations:
        dones = terminations
    else:
        dones = T.logical_or(terminations, truncations)

    for t in reversed(range(timesteps)):
        advantage = td_errors[t] + gamma * gae_lambda * advantage * T.logical_not(dones[t])
        advantages[t] = advantage
    return advantages

def compute_advantages_and_returns(
    rewards:T.Tensor,
    values:T.Tensor,
    next_values:T.Tensor,
    terminations:T.Tensor,
    truncations:T.Tensor,
    gamma:float,
    gae_lambda:float,
    bootstrap_truncations: bool,
    device:T.device|str|None = None
) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
    """
    Compute advantages and returns for a batch of trajectories.

    Args:
        rewards: Tensor of rewards [batch_size, num_envs].
        values: Tensor of values [batch_size, num_envs].
        next_values: Tensor of next values [batch_size, num_envs].
        terminations: Tensor of termination flags [batch_size, num_envs].
        truncations: Tensor of truncation flags [batch_size, num_envs].
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.
        bootstrap_truncations: Whether to bootstrap the returns on truncated episodes.
        device: Device for tensor operations.

    Returns:
        Tensor of advantages [batch_size, num_envs].
        Tensor of returns [batch_size, num_envs].
        Tensor of TD errors [batch_size, num_envs].
    """
    device = get_device(device)
    td_errors = compute_td_error(rewards, values, next_values, terminations, truncations, gamma, bootstrap_truncations)
    advantages = compute_gae(td_errors, terminations, truncations, gamma, gae_lambda, bootstrap_truncations, device)
    returns = advantages + values
    return advantages, returns, td_errors

def grad_norm_from_optimizer(optimizer: Optimizer) -> float:
    total_sq = None
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            grad_sq = p.grad.detach().pow(2).sum()
            total_sq = grad_sq if total_sq is None else total_sq + grad_sq
    return float(T.sqrt(total_sq)) if total_sq is not None else 0.0

def load_agent(config_dir:str | Path, load_weights: bool = True):
    """
    Load an agent from a configuration file.
    
    Args:
        config_dir: Path to the configuration directory
        load_weights: Whether to load the model weights
        
    Returns:
        The loaded agent
    """
    config = json.load(open(Path(config_dir) / 'config.json'))
    agent_type = config.get("agent_type")
    if agent_type is None:
        raise ValueError("agent_type must be specified in config")
        
    agent_class = get_agent_class_from_type(agent_type)
    if agent_class is None:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    agent = agent_class.load(config_dir, load_weights)
        
    return agent

def get_agent_class_from_type(agent_type: str):
    """
    Get the agent class from its type name.
    
    Args:
        agent_type: The type name of the agent
        
    Returns:
        The agent class
    """
    from .rl_agents import PPO, DDPG, Reinforce, ActorCritic, TD3, HER, SAC
    agent_classes = {
        "PPO": PPO,
        "DDPG": DDPG,
        "Reinforce": Reinforce,
        "ActorCritic": ActorCritic,
        "TD3": TD3,
        "HER": HER,
        "SAC": SAC
    }
    return agent_classes.get(agent_type) 

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