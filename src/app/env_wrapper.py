import sys
import os

from pydantic_core.core_schema import str_schema

# Use environment variable for IsaacLab path, with fallback to relative path
ISAACLAB_PATH = os.environ.get('ISAACLAB_PATH', os.path.join(os.path.dirname(__file__), '..', '..', 'IsaacLab', 'source'))
ISAACLAB_TASKS_PATH = os.path.join(ISAACLAB_PATH, 'isaaclab_tasks')

sys.path.append(ISAACLAB_PATH)
sys.path.append(ISAACLAB_TASKS_PATH)

import json
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Dict, List
from abc import ABC, abstractmethod
from collections import deque
import numpy as np
import torch as T

import gymnasium as gym
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message=".*Overriding environment.*already in registry.*")
    import gymnasium_robotics
from gymnasium.envs.registration import EnvSpec, WrapperSpec
import gymnasium.wrappers as gym_wrappers
import gymnasium.wrappers.vector as gym_vector_wrappers
from gymnasium.vector import VectorEnv, SyncVectorEnv, VectorWrapper, utils
import envpool

from .torch_utils import get_device
from .logging_config import get_logger
from .utils import to_torch, to_numpy
if TYPE_CHECKING:
    from .intrinsic_motivation import IntrinsicMotivation


@dataclass
class Observation:
    states: T.Tensor
    goals: T.Tensor | None = None
    ach_goals: T.Tensor | None = None
    rewards: T.Tensor | None = None
    intrinsic_rewards: T.Tensor | None = None
    terminations: T.Tensor | None = None
    truncations: T.Tensor | None = None
    n_step_trajectory: dict | None = None
    infos: dict | None = None

@dataclass
class Action:
    actions: T.Tensor
    raw_actions: T.Tensor | None = None
    log_probs: T.Tensor | None = None

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
    def __init__(
        self,
        env: VectorEnv,
        n: int,
        obs_key: str | None = None,
        goal_key: str | None = None,
        ach_goal_key: str | None = None,
        log_level: str = 'INFO',
        name: str | None = None,
        **kwargs
    ):
        super().__init__(env)
        self.name = name if name else self.__class__.__name__
        self.logger = get_logger(self.name, level=log_level.upper())
        self.kwargs = kwargs
        self.n = n
        self.obs_key = obs_key
        self.goal_key = goal_key
        self.ach_goal_key = ach_goal_key
        self.device = get_device()

        # Set internal attributes
        self._step = 0
        self._diag_freq = None
        self._log_diag = False
        self._nstep_diag_buffer = deque(maxlen=1024)
        if self.kwargs is not None:
            for key, value in self.kwargs.items():
                setattr(self, key, value)

        # Instantiate internal reference to Agent Intrinsic Motivation object
        # Set via set_intrinsic_motivation call in Trainer._initialize_run
        self.intrinsic_motivation = None

        # Buffer pointers
        self.head = T.zeros(self.num_envs, dtype=T.long, device=self.device)
        self.length = T.zeros(self.num_envs, dtype=T.long, device=self.device)
        self.prev_done = T.zeros(self.num_envs, dtype=T.bool, device=self.device)

        # Buffer storage (shape/dtype inferred from the first real step)
        self._buf_states = None
        self._buf_actions = None
        self._buf_raw_actions = None
        self._buf_log_probs = None
        self._buf_rewards = None
        self._buf_intrinsic_rewards = None
        self._buf_next_states = None
        self._buf_terminations = None
        self._buf_truncations = None
        self._buf_state_ach_goals = None
        self._buf_next_state_ach_goals = None
        self._buf_desired_goals = None

        self.current_states = None
        self.current_action = None

        # Helper index tensors
        self._t_idx = T.arange(self.n, device=self.device)
        self._env_idx = T.arange(self.num_envs, device=self.device)
        self._env_idx_nx1 = self._env_idx.unsqueeze(1).expand(self.num_envs, self.n)

    def set_action(self, action: Action) -> None:
        """Sets the action data for the current step.

        Args:
            action: The action to set.
        """
        self.current_action = action

    def set_intrinsic_motivation(self, intrinsic_motivation: "IntrinsicMotivation") -> None:
        self.intrinsic_motivation = intrinsic_motivation
    
    def reset(self, **kwargs):
        states, infos = self.env.reset(**kwargs)
        self.head.zero_()
        self.length.zero_()
        self.prev_done.zero_()
        self.current_states = states
        infos.setdefault('n-step trajectory', {})
        return states, infos

    def _alloc_like(self, sample: T.Tensor) -> T.Tensor:
        """(num_envs, *tail) -> (num_envs, n, *tail) pre-allocated buffer."""
        tail = tuple(sample.shape[1:])
        return T.zeros((self.num_envs, self.n, *tail), dtype=sample.dtype, device=self.device)

    def step(self, actions: T.Tensor):
        self._step += 1
        next_states, rewards, terminations, truncations, infos = self.env.step(actions)

        # Ensure tensors
        actions = T.as_tensor(actions, device=self.device)
        raw_actions = T.as_tensor(self.current_action.raw_actions, device=self.device) if self.current_action.raw_actions is not None else T.zeros_like(actions)
        rewards = T.as_tensor(rewards, device=self.device)
        log_probs = T.as_tensor(self.current_action.log_probs, device=self.device) if self.current_action.log_probs is not None else T.zeros_like(rewards)
        terminations = T.as_tensor(terminations, device=self.device)
        truncations = T.as_tensor(truncations, device=self.device)
        dones = T.logical_or(terminations, truncations)

        state_b = self.current_states[self.obs_key] if self.obs_key is not None else self.current_states
        next_state_b = next_states[self.obs_key] if self.obs_key is not None else next_states
        state_b = T.as_tensor(state_b, device=self.device)
        next_state_b = T.as_tensor(next_state_b, device=self.device)
        if self.goal_key:
            goals_b = T.as_tensor(self.current_states[self.goal_key], device=self.device)
            ach_goals_b = T.as_tensor(self.current_states[self.ach_goal_key], device=self.device)
            next_ach_goals_b = T.as_tensor(next_states[self.ach_goal_key], device=self.device)

        if self.intrinsic_motivation is not None:
            with T.no_grad():
                intrinsic_rewards = self.intrinsic_motivation.compute_rollout_reward(
                    state_b, next_state_b, actions, env_indices = T.arange(self.env.num_envs, device=self.intrinsic_motivation.device)
                )
            intrinsic_rewards = T.as_tensor(intrinsic_rewards, device=self.device).float()
            if intrinsic_rewards.ndim > 1:
                intrinsic_rewards = intrinsic_rewards.view(self.num_envs)
        else:
            intrinsic_rewards = T.zeros_like(rewards)
        
        # If first-time allocation (buf = None) set shapes from tensors
        if self._buf_states is None:
            self._buf_states = self._alloc_like(state_b)
            self._buf_next_states = self._alloc_like(next_state_b)
            self._buf_actions = self._alloc_like(actions)
            self._buf_raw_actions = self._alloc_like(actions)
            self._buf_log_probs = self._alloc_like(log_probs)
            self._buf_rewards = self._alloc_like(rewards)
            self._buf_intrinsic_rewards = self._alloc_like(intrinsic_rewards)
            self._buf_terminations = T.zeros((self.num_envs, self.n), dtype=terminations.dtype, device=self.device)
            self._buf_truncations = T.zeros((self.num_envs, self.n), dtype=truncations.dtype,  device=self.device)
            if self.goal_key:
                self._buf_state_ach_goals = self._alloc_like(ach_goals_b)
                self._buf_next_state_ach_goals = self._alloc_like(next_ach_goals_b)
                self._buf_desired_goals = self._alloc_like(goals_b)

        # Only envs whose previous step was NOT terminal get a new entry appended.
        active = ~self.prev_done
        write_pos = self.head

        env_idx = self._env_idx
        self._buf_states[env_idx, write_pos] = state_b
        self._buf_next_states[env_idx, write_pos] = next_state_b
        self._buf_actions[env_idx, write_pos] = actions
        self._buf_raw_actions[env_idx, write_pos] = raw_actions
        self._buf_log_probs[env_idx, write_pos] = log_probs
        self._buf_rewards[env_idx, write_pos] = rewards
        self._buf_intrinsic_rewards[env_idx, write_pos] = intrinsic_rewards
        self._buf_terminations[env_idx, write_pos] = terminations
        self._buf_truncations[env_idx, write_pos] = truncations
        if self.goal_key:
            self._buf_state_ach_goals[env_idx, write_pos] = ach_goals_b
            self._buf_next_state_ach_goals[env_idx, write_pos] = next_ach_goals_b
            self._buf_desired_goals[env_idx, write_pos] = goals_b

        # Advance pointer / length only for envs that actually appended.
        self.head = T.where(active, (self.head + 1) % self.n, self.head)
        self.length = T.where(active, T.clamp(self.length + 1, max=self.n), self.length)

        # Build the batched trajectory (valid envs only)
        trajectory = self._build_trajectories(dones=dones)
        infos['n-step trajectory'] = trajectory

        # Clear on done
        self.head = T.where(dones, T.zeros_like(self.head), self.head)
        self.length = T.where(dones, T.zeros_like(self.length), self.length)

        self.prev_done = dones
        self.current_states = next_states

        # Log diag values if diag
        if self._diag_freq is not None:
            self._log_diag = (self._step % self._diag_freq == 0)
        else:
            self._log_diag = False

        return next_states, rewards, terminations, truncations, infos

    def _build_trajectories(self, dones: T.Tensor | None = None):
        """
        Produces tensors of shape (num_valid_envs, n, *) where num_valid_envs is the
        number of envs with length > 0.
          - pad_mode "repeat" (states / next_states / actions): positions beyond
            `length` repeat the most recent valid entry.
          - pad_mode 0 (rewards / terminations / truncations): zero-filled.
        """
        valid = self.length > 0
        if not bool(valid.any()):
            return None

        n = self.n
        t = self._t_idx
        length = self.length
        head = self.head
        start = (head - length) % n

        # (num_envs, n) — True where this slot holds a real entry
        valid_mask = t.unsqueeze(0) < length.unsqueeze(1)

        # For "repeat" padding, map indices past `length` to the last real slot.
        last_valid = T.clamp(length - 1, min=0)
        t_for_gather = T.where(
            valid_mask,
            t.unsqueeze(0).expand(self.num_envs, n),
            last_valid.unsqueeze(1).expand(self.num_envs, n),
        )
        gather_idx = (start.unsqueeze(1) + t_for_gather) % n

        env_idx = self._env_idx_nx1

        # Repeat-padded
        states_all = self._buf_states[env_idx, gather_idx]
        next_states_all = self._buf_next_states[env_idx, gather_idx]
        actions_all = self._buf_actions[env_idx, gather_idx]
        raw_actions_all = self._buf_raw_actions[env_idx, gather_idx]
        log_probs_all = self._buf_log_probs[env_idx, gather_idx]

        # Zero-padded: gather first, then mask out invalid positions
        rewards_all = self._buf_rewards[env_idx, gather_idx]
        terminations_all = self._buf_terminations[env_idx, gather_idx]
        truncations_all = self._buf_truncations[env_idx, gather_idx]
        rewards_all = T.where(valid_mask, rewards_all, T.zeros_like(rewards_all))
        terminations_all = T.where(valid_mask, terminations_all, T.zeros_like(terminations_all))
        truncations_all = T.where(valid_mask, truncations_all, T.zeros_like(truncations_all))

        if self._buf_intrinsic_rewards is not None:
            intrinsic_rewards_all = self._buf_intrinsic_rewards[env_idx, gather_idx]
            intrinsic_rewards_all = T.where(valid_mask, intrinsic_rewards_all,
                                            T.zeros_like(intrinsic_rewards_all))
        else:
            intrinsic_rewards_all = T.zeros_like(rewards_all)
        
        if self.goal_key:
            state_ach_goals_all = self._buf_state_ach_goals[env_idx, gather_idx]
            next_state_ach_goals_all = self._buf_next_state_ach_goals[env_idx, gather_idx]
            desired_goals_all = self._buf_desired_goals[env_idx, gather_idx]

        # Filter to envs with data — one boolean-mask slice per tensor.
        trajectory = {
            'states': states_all[valid],
            'actions': actions_all[valid],
            'rewards': rewards_all[valid],
            'next_states': next_states_all[valid],
            'terminations': terminations_all[valid],
            'truncations': truncations_all[valid],
            'raw_actions': raw_actions_all[valid],
            'log_probs': log_probs_all[valid],
            'intrinsic_rewards': intrinsic_rewards_all[valid],
            'trajectory_lengths': length[valid],
        }
        if self.goal_key:
            trajectory['state_achieved_goals'] = state_ach_goals_all[valid]
            trajectory['next_state_achieved_goals'] = next_state_ach_goals_all[valid]
            trajectory['desired_goals'] = desired_goals_all[valid]

        if dones is not None and bool(dones.any()):
            # Fields that use "repeat" padding past the valid length.
            rep_fields = {
                'states': states_all,
                'next_states': next_states_all,
                'actions': actions_all,
                'raw_actions': raw_actions_all,
                'log_probs': log_probs_all,
            }
            # Fields that use zero padding past the valid length.
            zero_fields = {
                'rewards': rewards_all,
                'terminations': terminations_all,
                'truncations': truncations_all,
                'intrinsic_rewards': intrinsic_rewards_all,
            }
            if self.goal_key:
                rep_fields['state_achieved_goals'] = state_ach_goals_all
                rep_fields['next_state_achieved_goals'] = next_state_ach_goals_all
                rep_fields['desired_goals'] = desired_goals_all

            # Flush tails if there is a terminal state so each step in the n-step trajectory
            # becomes an anchor of its own n-step window
            tail_rows: dict[str, list[T.Tensor]] = {
                k: [] for k in (*rep_fields, *zero_fields)
            }
            tail_lengths: list[int] = []
            t = self._t_idx  # arange(n)
            for e in dones.nonzero(as_tuple=True)[0].tolist():
                L = int(length[e].item())
                # j = 1 .. L-1 peels off the oldest entry each time.
                for j in range(1, L):
                    new_len = L - j
                    # Shift left by j; clamp keeps us in-bounds (the clamped
                    # tail lands on already-padded slots for repeat fields).
                    idx = T.clamp(t + j, max=n - 1)
                    new_valid = t < new_len  # bool [n]
                    for key, arr in rep_fields.items():
                        tail_rows[key].append(arr[e, idx])
                    for key, arr in zero_fields.items():
                        row = arr[e, idx].clone()
                        # Re-zero padding: required when L == n, where the
                        # clamped indices would otherwise repeat a real entry
                        # (e.g. the terminal reward/flag) into padding slots.
                        row[~new_valid] = 0
                        tail_rows[key].append(row)
                    tail_lengths.append(new_len)
            if tail_lengths:
                for key in tail_rows:
                    stacked = T.stack(tail_rows[key], dim=0)  # [num_tail, n, ...]
                    trajectory[key] = T.cat([trajectory[key], stacked], dim=0)
                trajectory['trajectory_lengths'] = T.cat(
                    [
                        trajectory['trajectory_lengths'],
                        T.tensor(tail_lengths, dtype=length.dtype, device=self.device),
                    ],
                    dim=0,
                )

        # Collect n-step boundary diagnostics
        if self._nstep_diag_buffer is not None:
            valid_lengths = length[valid].tolist()
            had_term = (terminations_all[valid] | truncations_all[valid]).any(dim=1)
            for i, L in enumerate(valid_lengths):
                self._nstep_diag_buffer.append((int(L), bool(had_term[i])))
        
        return trajectory

    def get_nstep_diagnostics(self) -> dict:
        """Return and clear accumulated n-step window statistics."""
        if not self._nstep_diag_buffer:
            return {}

        lengths = []
        short_after_term = 0
        total = len(self._nstep_diag_buffer)

        for length, had_term in self._nstep_diag_buffer:
            lengths.append(length)
            if had_term and length < self.n:
                short_after_term += 1

        self._nstep_diag_buffer.clear()

        return {
            "nstep/avg_trajectory_length": float(sum(lengths) / len(lengths)) if lengths else 0.0,
            "nstep/pct_short_windows_after_term": (short_after_term / total) if total > 0 else 0.0,
        }

class OneHotObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.observation_space, gym.spaces.Discrete), "Observation space must be Discrete."
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.observation_space.n,), dtype=np.float32)
    
    def observation(self, obs):
        one_hot = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        one_hot[obs] = 1.0
        return one_hot

class NumpyToTorch(VectorWrapper):
    def __init__(self, env, device=None):
        super().__init__(env)
        self.device = device
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=to_numpy(options))
        return to_torch(obs, self.device), to_torch(info, self.device)
    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(to_numpy(actions))
        return (
            to_torch(obs, self.device),
            to_torch(reward, self.device),
            to_torch(terminated, self.device),
            to_torch(truncated, self.device),
            to_torch(info, self.device),
        )
    def render(self):
        return self.env.render()

class VectorOneHotObservation(VectorWrapper):
    """Vectorized one-hot encoding for Discrete observation spaces."""
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.single_observation_space, gym.spaces.Discrete)
        self._n = self.single_observation_space.n

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._encode(obs), info

    def step(self, actions):
        obs, rew, term, trunc, info = self.env.step(actions)
        return self._encode(obs), rew, term, trunc, info

    def _encode(self, obs):
        batch = obs.shape[0] if hasattr(obs, 'shape') else len(obs)
        one_hot = np.zeros((batch, self._n), dtype=np.float32)
        one_hot[np.arange(batch), obs.astype(int).ravel()] = 1.0
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
    },
    "VectorOneHotObservation": {
    "cls": VectorOneHotObservation,
    "vector_aware": True,
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

class EnvWrapper:
    """
    Abstract base class for environment wrappers.

    This class defines the required interface for custom environment wrappers.
    """

    def __init__(
        self,
        cfg:str,
        num_envs:int=1,
        obs_key:str|None=None,
        goal_key:str|None=None,
        ach_goal_key:str|None=None,
        wrappers:list[dict]|None=None,
        render_mode:str|None=None,
        seed:int|None=None
    ):
        self.env_id = cfg
        self.num_envs = num_envs
        self.obs_key = obs_key
        self.goal_key = goal_key
        self.ach_goal_key = ach_goal_key
        self.wrappers = wrappers
        self.render_mode = render_mode
        if seed is None:
            seed = T.randint(2**31-1, (1,)).item()
        self.seed = seed

    def extract_states_goals(
        self,
        states: np.ndarray | T.Tensor | dict | list[dict]
    )->tuple[T.Tensor, T.Tensor | None, T.Tensor | None]:
        """Extract the states and goals from the passed states argument and returns them as Tensors.
        
        Args:
            states (np.ndarray | T.Tensor | dict | list[dict]): States to extract from.
        
        Returns:
            tuple: Tuple of states, goals, and achieved goals as Tensors.
        """
        device = get_device()
        if isinstance(states, list):
            obs_list = []
            goals_list = []
            ach_goals_list = []
            for step_data in states:
                if isinstance(step_data, dict):
                    if not self.obs_key:
                        raise ValueError("Goal-aware observation spaces require obs_key to be set")
                    step_obs = step_data.get(self.obs_key)
                    if self.goal_key:
                        step_goal = step_data.get(self.goal_key)
                    else:
                        step_goal = None
                    if self.ach_goal_key:
                        step_ach_goal = step_data.get(self.ach_goal_key)
                    else:
                        step_ach_goal = None
                else:
                    break

                # Convert to tensor if needed
                if not isinstance(step_obs, T.Tensor):
                    step_obs = T.tensor(step_obs, dtype=T.float32, device=device)
                if self.goal_key and step_goal is not None:
                    if not isinstance(step_goal, T.Tensor):
                        step_goal = T.tensor(step_goal, dtype=T.float32, device=device)
                    goals_list.append(step_goal)
                if self.ach_goal_key and step_ach_goal is not None:
                    if not isinstance(step_ach_goal, T.Tensor):
                        step_ach_goal = T.tensor(step_ach_goal, dtype=T.float32, device=device)
                    ach_goals_list.append(step_ach_goal)
                obs_list.append(step_obs)
            obs = T.stack(obs_list, dim=0)
            
            if self.goal_key:
                goals = T.stack(goals_list, dim=0)
            else:
                goals = None
            if self.ach_goal_key:
                ach_goals = T.stack(ach_goals_list, dim=0)
            else:
                ach_goals = None

        elif isinstance(states, dict):
            if not self.obs_key:
                raise ValueError("Goal-aware observation spaces require obs_key to be set")
            obs = states.get(self.obs_key)
            if self.goal_key:
                goals = states.get(self.goal_key)
            else:
                goals = None
            if self.ach_goal_key:
                ach_goals = states.get(self.ach_goal_key)
            else:
                ach_goals = None
        else:
            obs = states
            goals = None
            ach_goals = None

        if not isinstance(obs, T.Tensor):
            obs = T.tensor(obs, dtype=T.float32, device=device)
        if goals is not None and not isinstance(goals, T.Tensor):
            goals = T.tensor(goals, dtype=T.float32, device=device)
        if ach_goals is not None and not isinstance(ach_goals, T.Tensor):
            ach_goals = T.tensor(ach_goals, dtype=T.float32, device=device)
        
        return obs, goals, ach_goals

    def _find_nstep_wrapper(self) -> VectorNStepReward | None:
        """Finds the VectorNStepReward wrapper in the environment chain."""
        env = self
        while env is not None:
            if isinstance(env, VectorNStepReward):
                return env
            env = getattr(env, 'env', None)

    def get_base_env(self):
        """Recursively unwrap an environment to get the base environment."""
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env

    @property
    def config(self):
        """
        Get the configuration of the wrapper.

        Returns:
            dict: Configuration dictionary.
        """
        return {
            "type": self.__class__.__name__,
            "config":{
                "cfg": self.env_id,
                "num_envs": self.num_envs,
                "obs_key": self.obs_key,
                "goal_key": self.goal_key,
                "ach_goal_key": self.ach_goal_key,
                "wrappers": self.wrappers,
                "render_mode": self.render_mode,
                "seed": self.seed,
            }
        }

    @abstractmethod
    def reset(self):
        """
        Reset the environment to an initial state.

        Returns:
            Any: Initial observation of the environment.
        """
        pass
    
    @abstractmethod
    def step(self, action) -> Observation:
        """
        Take an action in the environment.

        Args:
            action: The action to be taken.

        Returns:
            Observation: A dataclass containing the current state, transition state, rewards, terminations, truncations, additional info, current goals, transition goals, current achieved goals, transition achieved goals.
        """
        pass

    @abstractmethod
    def _initialize_env(self):
        """
        Initialize the environment.

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
            elif config['type'] == 'envpool':
                return EnvPoolWrapper.from_json(json_string)
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
    def __init__(
        self,
        cfg:str,
        num_envs:int=1,
        obs_key:str|None=None,
        goal_key:str|None=None,
        ach_goal_key:str|None=None,
        wrappers:list[dict]|None=None,
        render_mode:str|None=None,
        seed:int|None=None
    ):
        super().__init__(cfg, num_envs, obs_key, goal_key, ach_goal_key, wrappers, render_mode, seed)
        # self.env_id = cfg
        # self.num_envs = num_envs
        # self.obs_key = obs_key
        # self.goal_key = goal_key
        # self.ach_goal_key = ach_goal_key
        # self.wrappers = wrappers
        # self.render_mode = render_mode
        # if seed is None:
        #     seed = T.randint(2**31-1, (1,)).item()
        # self.seed = seed
        self.env = self._initialize_env()
        

    def _initialize_env(self):
        """
        Initialize the Gymnasium environments.

        
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
            vector_kwargs={"autoreset_mode": "NextStep"},
            wrappers=single_wrappers,
            render_mode=self.render_mode
        )

        # Apply vector-aware wrappers to the entire vec_env
        for cls, params in vector_wrappers:
            vec_env = cls(vec_env, **params)

        # Wrap vectorized environment to return tensors
        vec_env = NumpyToTorch(vec_env, device=get_device())

        return vec_env

    def render_frame(self)->np.ndarray:
        """Renders a frame from the environment.
        
        Returns:
            np.ndarray: The rendered frame.
        """
        frame = self.env.render()        
        return frame[0]
        

    def reset(self, seed:int|None=None):
        if seed is not None:
            effective_seed = seed
        else:
            effective_seed = self.seed

        states, infos = self.env.reset(seed=effective_seed)
        self.env.action_space.seed(seed=effective_seed)
        
        obs, goals, ach_goals = self.extract_states_goals(states)

        observation = Observation(
            states=obs,
            goals=goals,
            ach_goals=ach_goals,
            infos=infos
        )

        if 'n-step trajectory' in infos:
            observation.n_step_trajectory = infos.pop('n-step trajectory')
        return observation

    def step(self, action)->Observation:
        states, rewards, terminations, truncations, infos = self.env.step(action)

        # Separate observations, goals, and achieved goals 
        obs, goals, ach_goals = self.extract_states_goals(states)

        observation = Observation(
            states=obs,
            goals=goals,
            ach_goals=ach_goals,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            infos=infos
        )

        if 'n-step trajectory' in infos:
            observation.n_step_trajectory = infos.pop('n-step trajectory')
        return observation

    def sample_observation(self):
        actions = self.action_space.sample()
        return self.step(actions)
    
    def format_actions(self, actions: np.ndarray | T.Tensor):
        if isinstance(actions, T.Tensor):
            actions = actions.cpu().numpy()
        if isinstance(self.action_space, gym.spaces.Box):
            num_envs = self.env.num_envs
            num_actions = self.action_space.shape[-1]
            return actions.reshape(num_envs, num_actions)
        if isinstance(self.action_space, gym.spaces.Discrete) or isinstance(self.action_space, gym.spaces.MultiDiscrete):
            return actions.ravel()
    
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
    def finite_horizon(self)->bool:
        """
        Returns True if the environment has a finite horizon.
        Finite horizon is determined by checking if the base environment spec contains has
        a max_episode_steps attribute that is not None, or if the environment is wrapped in a 
        TimeLimit wrapper.
        """
        base_env = self.get_base_env()
        if hasattr(base_env, 'spec') and base_env.spec is not None:
            return base_env.spec.max_episode_steps is not None

        env = self.env
        while hasattr(env, 'env'):
            if isinstance(env, gym.wrappers.TimeLimit):
                return True
            env = env.env
        
        return False
    
    @property
    def config(self):
        """
        Get the configuration of the wrapper.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().config
        config['type'] = "gymnasium"
        return config
        # return {
        #     "type": "gymnasium",
        #     "config":{
        #         "cfg": self.env_id,
        #         "num_envs": self.num_envs,
        #         "wrappers": self.wrappers,
        #         "render_mode": self.render_mode,
        #         "seed": self.seed,
        #         "obs_key": self.obs_key,
        #         "goal_key": self.goal_key,
        #         "ach_goal_key": self.ach_goal_key,
        #     }
        # }
    
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
        try:
            return cls(**config)
        except Exception as e:
            raise ValueError(f"Environment wrapper error: {config}, {e}")

class EnvPoolAdapter(VectorEnv):
    """Adapts an EnvPool gymnasium env to be compatible with the VectorWrapper chain."""
    def __init__(self, envpool_env, num_envs: int):
        self._env = envpool_env
        self.num_envs = num_envs
        self.single_observation_space = envpool_env.observation_space
        self.single_action_space = envpool_env.action_space
        self.observation_space = utils.batch_space(envpool_env.observation_space, num_envs)
        self.action_space = utils.batch_space(envpool_env.action_space, num_envs)

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset()
        return obs, info

    def step(self, actions):
        return self._env.step(actions)

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        self._env.close()

    @property
    def spec(self):
        return getattr(self._env, 'spec', None)

class EnvPoolWrapper(EnvWrapper):
    WRAPPER_TO_ENVPOOL_PARAM = {
        "AtariPreprocessing": lambda p: {
            "frame_skip": p.get("frame_skip", 4),
            "img_height": 84,
            "img_width": 84,
        },
        "FrameStackObservation": lambda p: {
            "stack_num": p.get("stack_size", 4),
        },
        "TimeLimit": lambda p: {
            "max_episode_steps": p.get("max_episode_steps", 1000),
        },
        "ResizeObservation": lambda p: {
            "img_height": p.get("shape", 84) if isinstance(p.get("shape", 84), int) else p["shape"][0],
            "img_width": p.get("shape", 84) if isinstance(p.get("shape", 84), int) else p["shape"][1],
        },
    }

    def __init__(
        self,
        cfg: str,
        num_envs: int = 1,
        obs_key: str | None = None,
        goal_key: str | None = None,
        ach_goal_key: str | None = None,
        num_threads: int | None = None,
        wrappers: list[dict] | None = None,
        render_mode: str | None = None,
        seed: int | None = None
    ):
        super().__init__(cfg, num_envs, obs_key, goal_key, ach_goal_key, wrappers, render_mode, seed)
        # self.env_id = cfg
        # self.num_envs = num_envs
        # self.obs_key = obs_key
        # self.goal_key = goal_key
        # self.ach_goal_key = ach_goal_key
        # self.wrappers = wrappers
        # self.render_mode = render_mode
        # if seed is None:
        #     seed = T.randint(2**31 - 1, (1,)).item()
        # self.seed = seed
        self.num_threads = num_threads
        self.env = self._initialize_env()

    def _initialize_env(self):
        envpool_kwargs = {
            "task_id": self.env_id,
            "num_envs": self.num_envs,
            "seed": self.seed,
        }
        if self.render_mode:
            envpool_kwargs["render_mode"] = self.render_mode
        if self.num_threads:
            envpool_kwargs["num_threads"] = self.num_threads
        else:
            envpool_kwargs["num_threads"] = self.num_envs

        vector_wrappers = []
        if self.wrappers:
            for wrapper in self.wrappers:
                wtype = wrapper.get("type", "")
                params = wrapper.get("params", {})

                if wtype in self.WRAPPER_TO_ENVPOOL_PARAM:
                    envpool_kwargs.update(self.WRAPPER_TO_ENVPOOL_PARAM[wtype](params))
                elif wtype in WRAPPER_REGISTRY and WRAPPER_REGISTRY[wtype].get("vector_aware"):
                    default_p = WRAPPER_REGISTRY[wtype]["default_params"].copy()
                    default_p.update(params)
                    vector_wrappers.append((WRAPPER_REGISTRY[wtype]["cls"], default_p))
                elif hasattr(gym_vector_wrappers, wtype):
                    cls = getattr(gym_vector_wrappers, wtype)
                    vector_wrappers.append((cls, params))
                else:
                    raise ValueError(
                        f"Wrapper '{wtype}' is not supported by EnvPool. "
                        f"Use a vector-aware wrapper or map it to an envpool parameter."
                    )

        raw_env = envpool.make_gymnasium(**envpool_kwargs)
        env = EnvPoolAdapter(raw_env, num_envs=self.num_envs)

        for cls, params in vector_wrappers:
            env = cls(env, **params)

        env = NumpyToTorch(env, device=get_device())
        return env

    def extract_states_goals(
        self,
        states: np.ndarray | T.Tensor | dict | list[dict]
    )->tuple[T.Tensor, T.Tensor | None, T.Tensor | None]:
        """Extract the states and goals from the passed states argument and returns them as Tensors.
        
        Args:
            states (np.ndarray | T.Tensor | dict | list[dict]): States to extract from.
        
        Returns:
            tuple: Tuple of states, goals, and achieved goals as Tensors.
        """
        device = get_device()
        if isinstance(states, list):
            obs_list = []
            goals_list = []
            ach_goals_list = []
            for step_data in states:
                if isinstance(step_data, dict):
                    if not self.obs_key:
                        raise ValueError("Goal-aware observation spaces require obs_key to be set")
                    step_obs = step_data.get(self.obs_key)
                    if self.goal_key:
                        step_goal = step_data.get(self.goal_key)
                    else:
                        step_goal = None
                    if self.ach_goal_key:
                        step_ach_goal = step_data.get(self.ach_goal_key)
                    else:
                        step_ach_goal = None
                else:
                    break

                # Convert to tensor if needed
                if not isinstance(step_obs, T.Tensor):
                    step_obs = T.tensor(step_obs, dtype=T.float32, device=device)
                if self.goal_key and step_goal is not None:
                    if not isinstance(step_goal, T.Tensor):
                        step_goal = T.tensor(step_goal, dtype=T.float32, device=device)
                    goals_list.append(step_goal)
                if self.ach_goal_key and step_ach_goal is not None:
                    if not isinstance(step_ach_goal, T.Tensor):
                        step_ach_goal = T.tensor(step_ach_goal, dtype=T.float32, device=device)
                    ach_goals_list.append(step_ach_goal)
                obs_list.append(step_obs)
            obs = T.stack(obs_list, dim=0)
            
            if self.goal_key:
                goals = T.stack(goals_list, dim=0)
            else:
                goals = None
            if self.ach_goal_key:
                ach_goals = T.stack(ach_goals_list, dim=0)
            else:
                ach_goals = None

        elif isinstance(states, dict):
            if not self.obs_key:
                raise ValueError("Goal-aware observation spaces require obs_key to be set")
            obs = states.get(self.obs_key)
            if self.goal_key:
                goals = states.get(self.goal_key)
            else:
                goals = None
            if self.ach_goal_key:
                ach_goals = states.get(self.ach_goal_key)
            else:
                ach_goals = None
        else:
            obs = states
            goals = None
            ach_goals = None

        if not isinstance(obs, T.Tensor):
            obs = T.tensor(obs, dtype=T.float32, device=device)
        if goals is not None and not isinstance(goals, T.Tensor):
            goals = T.tensor(goals, dtype=T.float32, device=device)
        if ach_goals is not None and not isinstance(ach_goals, T.Tensor):
            ach_goals = T.tensor(ach_goals, dtype=T.float32, device=device)
        
        return obs, goals, ach_goals

    def render_frame(self)->np.ndarray:
        """Renders a frame from the environment.
        
        Returns:
            np.ndarray: The rendered frame.
        """
        frame = self.env.render()        
        return frame[0]
        
    def reset(self, seed:int|None=None):
        if seed is not None:
            effective_seed = seed
        else:
            effective_seed = self.seed

        states, infos = self.env.reset(seed=effective_seed)
        self.env.action_space.seed(seed=effective_seed)
        
        obs, goals, ach_goals = self.extract_states_goals(states)

        observation = Observation(
            states=obs,
            goals=goals,
            ach_goals=ach_goals,
            infos=infos
        )

        if 'n-step trajectory' in infos:
            observation.n_step_trajectory = infos.pop('n-step trajectory')
        return observation

    def step(self, action)->Observation:
        states, rewards, terminations, truncations, infos = self.env.step(action)

        # Separate observations, goals, and achieved goals 
        obs, goals, ach_goals = self.extract_states_goals(states)

        observation = Observation(
            states=obs,
            goals=goals,
            ach_goals=ach_goals,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            infos=infos
        )

        if 'n-step trajectory' in infos:
            observation.n_step_trajectory = infos.pop('n-step trajectory')
        return observation

    def sample_observation(self):
        actions = self.action_space.sample()
        observation = self.step(actions)
        obs, goals, ach_goals = self.extract_states_goals(observation.states)
        return Observation(
            states=obs,
            goals=goals,
            ach_goals=ach_goals
        )
    
    def format_actions(self, actions: np.ndarray | T.Tensor):
        if isinstance(actions, T.Tensor):
            actions = actions.cpu().numpy()
        if isinstance(self.action_space, gym.spaces.Box):
            num_envs = self.env.num_envs
            num_actions = self.action_space.shape[-1]
            return actions.reshape(num_envs, num_actions)
        if isinstance(self.action_space, gym.spaces.Discrete) or isinstance(self.action_space, gym.spaces.MultiDiscrete):
            return actions.ravel()
    
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
    def finite_horizon(self) -> bool:
        spec = getattr(self.env, 'spec', None)
        if spec and hasattr(spec, 'max_episode_steps'):
            return spec.max_episode_steps is not None
        return False
    
    @property
    def config(self):
        """
        Get the configuration of the wrapper.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().config
        config['type'] = "envpool"
        config['config']['num_threads'] = self.num_threads
        return config
        # return {
        #     "type": "envpool",
        #     "config":{
        #         "cfg": self.env_id,
        #         "num_envs": self.num_envs,
        #         "wrappers": self.wrappers,
        #         "render_mode": self.render_mode,
        #         "seed": self.seed,
        #         "obs_key": self.obs_key,
        #         "goal_key": self.goal_key,
        #         "ach_goal_key": self.ach_goal_key,
        #         "num_threads": self.num_threads,
        #     }
        # }
    
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
        Create a EnvPool wrapper instance from a JSON string.

        Args:
            json_env_spec (str): JSON string representing the configuration.

        Returns:
            EnvPoolWrapper: A new EnvPool wrapper instance.
        """
        config = json.loads(json_env_spec)
        config = config['config']
        try:
            return cls(**config)
        except Exception as e:
            raise ValueError(f"Environment wrapper error: {config}, {e}")
    
class IsaacLabAdapter(VectorEnv):
    """Adapts an Isaac Lab ``ManagerBasedRLEnv`` to the gymnasium ``VectorWrapper``
    chain (e.g. :class:`VectorNStepReward`), which requires a ``VectorEnv``.

    ``ManagerBasedRLEnv`` is vectorized but deliberately does *not* inherit from
    ``gymnasium.vector.VectorEnv``. It does already expose ``num_envs`` and
    goal-conditioned ``Dict`` observation spaces (one key per observation group),
    so this adapter just forwards ``reset``/``step`` and the spaces.

    It also supplies the goal reward used by Hindsight Experience Replay. Isaac
    Lab manager-based envs compute rewards through their ``RewardManager`` and
    expose no goal-conditioned reward function, so HER (which resolves
    ``compute_reward`` via :meth:`EnvWrapper.get_base_env`) needs one here. The
    online ``RewardManager`` term should match this sparse reward so collected
    and relabeled transitions share the same scale.
    """

    def __init__(self, env, distance_threshold: float = 0.05):
        self._env = env
        self.num_envs = env.num_envs
        self.single_observation_space = env.single_observation_space
        self.single_action_space = env.single_action_space
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.distance_threshold = distance_threshold

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed)

    def step(self, actions):
        return self._env.step(actions)

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        """Sparse goal reward: 0 if within ``distance_threshold`` else -1 (batched)."""
        d = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
        return -(d > self.distance_threshold).astype(np.float32)

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        self._env.close()

    @property
    def spec(self):
        return getattr(self._env, 'spec', None)


class IsaacSimWrapper(EnvWrapper):
    def __init__(
        self,
        cfg:str,
        num_envs:int=1,
        obs_key:str='policy',
        goal_key:str|None=None,
        ach_goal_key:str|None=None,
        wrappers:list[dict]|None=None,
        render_mode:str='headless',
        seed:int|None=None,
        distance_threshold:float=0.05,
    ):
        """
        Wrapper for Isaac Sim environments.

        This wrapper supports initialization, resetting, stepping, rendering,
        and JSON-based serialization of Isaac Sim environments.
        """
        super().__init__(cfg, num_envs, obs_key, goal_key, ach_goal_key, wrappers, render_mode, seed)
        # self.cfg = cfg
        # self.num_envs = num_envs
        # self.wrappers = wrappers
        # self.render_mode = render_mode
        # self.obs_key = obs_key
        # self.goal_key = goal_key
        # self.ach_goal_key = ach_goal_key
        # if seed is None:
        #     seed = np.random.randint(1000)
        # self.seed = seed
        self.distance_threshold = distance_threshold
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
            app_launcher = AppLauncher(headless=(self.render_mode=='headless'), device="cuda:0", enable_cameras=False)
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

        module_path, class_name = self.env_id.split(':')
        cfg_class = getattr(importlib.import_module(module_path), class_name)
        cfg = cfg_class()
        cfg.scene.num_envs = self.num_envs
        cfg.sim.device = "cuda:0"
        cfg.seed = self.seed
        env = ManagerBasedRLEnv(cfg=cfg)
        # Adapt to the gymnasium VectorWrapper chain (VectorNStepReward, etc.) and
        # supply the goal reward HER recomputes during relabeling.
        env = IsaacLabAdapter(env, distance_threshold=self.distance_threshold)
        if self.wrappers:
            for wrapper in self.wrappers:
                if wrapper['type'] in WRAPPER_REGISTRY:
                    default_params = WRAPPER_REGISTRY[wrapper['type']]["default_params"].copy()
                    override_params = wrapper.get("params", {})
                    final_params = {**default_params, **override_params}
                    # final_params.update({"obs_key": self.obs_key, "goal_key": self.goal_key})
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
    
    def reset(self, seed:int|None=None):
        if seed is not None:
            effective_seed = seed
        else:
            effective_seed = self.seed

        states, infos = self.env.reset(seed=effective_seed)
        self.env.action_space.seed(seed=effective_seed)
        
        obs, goals, ach_goals = self.extract_states_goals(states)

        observation = Observation(
            states=obs,
            goals=goals,
            ach_goals=ach_goals,
            infos=infos
        )

        if 'n-step trajectory' in infos:
            observation.n_step_trajectory = infos.pop('n-step trajectory')
        return observation

    def close(self):
        self.env.close()
        self.app.close()

    def step(self, action)->Observation:
        states, rewards, terminations, truncations, infos = self.env.step(action)

        # Separate observations, goals, and achieved goals 
        obs, goals, ach_goals = self.extract_states_goals(states)

        observation = Observation(
            states=obs,
            goals=goals,
            ach_goals=ach_goals,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            infos=infos
        )

        if 'n-step trajectory' in infos:
            observation.n_step_trajectory = infos.pop('n-step trajectory')
        return observation

    @property
    def config(self):
        config = super().config
        config['type'] = "isaacsim"
        config['config']['distance_threshold'] = self.distance_threshold
        return config
        # return {
        #     "type": "isaacsim",
        #     "config":{
        #         "cfg": self.cfg,
        #         "num_envs": self.num_envs,
        #         "wrappers": self.wrappers if self.wrappers else [],
        #         "render_mode": self.render_mode,
        #         "seed": self.seed,
        #         "obs_key": self.obs_key,
        #         "goal_key": self.goal_key,
        #     }
        # }

    def to_json(self):
        return json.dumps(self.config)

    @classmethod
    def from_json(cls, json_string):
        config = json.loads(json_string)
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

