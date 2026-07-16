from abc import abstractmethod
import torch as T
import numpy as np
import gymnasium as gym
from typing import Optional, Tuple, List, Any, Dict
from collections import defaultdict
import math

from .env_wrapper import Observation, Action, EnvWrapper, GymnasiumWrapper, IsaacSimWrapper
from .her import HindsightRelabeler, AchievedGoalPool
from .torch_utils import get_device


class SumTree:
    """
    Binary sum tree for priority-based sampling.
    """
 
    def __init__(self, capacity: int, device: T.device):
        if capacity <= 0:
            raise ValueError(f"SumTree capacity must be positive, got {capacity}")
 
        # Round up to next power of 2.
        self.capacity: int = 1 if capacity == 1 else 1 << (capacity - 1).bit_length()
        # depth = log2(capacity); how many levels to descend from root to a leaf.
        self.depth: int = self.capacity.bit_length() - 1
        self.device = get_device(device)
 
        # Tree has 2*capacity - 1 nodes.
        self.tree = T.zeros(2 * self.capacity - 1, dtype=T.float32, device=self.device)
        self.max_priority = T.tensor(1.0, dtype=T.float32, device=self.device)
 
    def update(self, data_indices: T.Tensor, priorities: T.Tensor) -> None:
        """Write priorities at data indices and propagate sums up to root."""
        # Floor priorities to avoid creating zero-probability leaves.
        priorities = T.clamp(priorities, min=1e-6)
        if priorities.numel() > 0:
            self.max_priority = T.maximum(self.max_priority, priorities.max())
 
        # Map data indices to tree leaf indices.
        indices = data_indices + (self.capacity - 1)
        self.tree[indices] = priorities
 
        # Propagate up.
        for _ in range(self.depth):
            parents = T.unique((indices - 1) // 2)
            parents = parents[parents >= 0]
            if parents.numel() == 0:
                break
            left = 2 * parents + 1
            right = 2 * parents + 2
            self.tree[parents] = self.tree[left] + self.tree[right]
            indices = parents
 
    def get(self, p_values: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        """Map cumulative-priority values to (data_index, leaf_priority) pairs."""
        tree_indices = self._traverse(p_values)
        # Defensive clamp — shouldn't be needed if the tree is well-formed.
        tree_indices = T.clamp(tree_indices, 0, self.tree.size(0) - 1)
        priorities = self.tree[tree_indices]
        data_indices = (tree_indices - (self.capacity - 1)).clamp(0, self.capacity - 1)
        return data_indices, priorities
 
    def _traverse(self, p_values: T.Tensor) -> T.Tensor:
        """
        Vectorized batch descent.
 
        At each level every batch element is at some node `idx`; read its
        left child's stored sum and decide left vs. right based on whether `p`
        fits in the left subtree.
        """
        idx = T.zeros_like(p_values, dtype=T.long)
        p = p_values.clone()
        tree_size = self.tree.size(0)
 
        for _ in range(self.depth):
            left = 2 * idx + 1
            in_bounds = left < tree_size  # guard against descending past leaves
            left_safe = T.where(in_bounds, left, idx)
            left_val = self.tree[left_safe]
            go_right = (p > left_val) & in_bounds
            p = T.where(go_right, p - left_val, p)
            idx = T.where(go_right, left_safe + 1, left_safe)
        return idx
 
    @property
    def total_priority(self) -> float:
        """Total priority lives at the root."""
        return self.tree[0].item()

class Buffer:
    """
    Base class for replay buffers with N-step functionality.
    """
    def __init__(
        self,
        env: EnvWrapper,
        buffer_size: int,
        hindsight: HindsightRelabeler | None = None,
        device: Optional[str] = None
    ):
        self.env = env
        self.buffer_size = buffer_size
        self.hindsight = hindsight
        if self.hindsight is not None:
            # Initialize episode trajectory buffers for each environment to
            # rebuild the trajectories with hindsight relabeling.
            self._ep_buffers = [self._empty_ep_buffer() for _ in range(env.num_envs)]
        self.device = get_device(device)
        self.env_steps = 0 # Tracks how many steps have been taken in the environment

        # Set observation, goal, and action space shapes
        if isinstance(self.env.single_observation_space, gym.spaces.Dict):
            self.obs_space_shape = self.env.single_observation_space[self.env.obs_key].shape
            if self.env.goal_key is not None:
                self.goal_space_shape = self.env.single_observation_space[self.env.goal_key].shape
            else:
                self.goal_space_shape = None
        else:
            self.obs_space_shape = self.env.single_observation_space.shape
            self.goal_space_shape = None

        if isinstance(self.env.single_action_space, gym.spaces.Box):
            self.action_type = T.float32
            self.action_space_shape = self.env.single_action_space.shape
        else: # Discrete
            self.action_type = T.long
            self.action_space_shape = (1,)

    def _empty_ep_buffer(self) -> Dict[str, List[T.Tensor]]:
        return {
            "states": [], "actions": [], "raw_actions": [], "log_probs": [],
            "rewards": [], "intrinsic_rewards": [],
            "next_states": [], "terminations": [], "truncations": [],
            "state_achieved_goals": [], "next_state_achieved_goals": [],
            "desired_goals": [], "trajectory_lengths": [],
        }

    @abstractmethod
    def add(self, states, actions, rewards, next_states, dones):
        """
        Add a transition to the buffer, including trajectory metadata.
        Abstract method to be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int) -> Tuple[T.Tensor, ...]:
        """
        Sample a batch of transitions from the buffer.
        Abstract method to be implemented by subclasses.
        """
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        return {
            'type': self.__class__.__name__,
            'config': {
                "env": self.env.to_json(),
                "buffer_size": self.buffer_size,
                "hindsight": self.hindsight.get_config() if self.hindsight is not None else None,
                "device": self.device.type
            }
        }

    @classmethod
    def create_instance(cls, buffer_type: str, **kwargs) -> 'Buffer':
        buffer_types = {
            "ReplayBuffer": ReplayBuffer,
            "PrioritizedReplayBuffer": PrioritizedReplayBuffer,
            "RolloutBuffer": RolloutBuffer,
            "TrajectoryBuffer": TrajectoryBuffer,
        }
        if buffer_type in buffer_types:
            return buffer_types[buffer_type](**kwargs)
        else:
            raise ValueError(f"{buffer_type} is not a subclass of Buffer")

    def save_state(self, path) -> None:
        """Persist the stored transitions + counters (optional resume data).

        Generic over subclasses: every tensor attribute and every int/float/bool
        counter is dumped. The SumTree of a prioritized buffer is handled
        explicitly since it is a nested object rather than a bare tensor.
        """
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors, scalars = {}, {}
        for key, value in self.__dict__.items():
            if isinstance(value, T.Tensor):
                tensors[key] = value.detach().cpu()
            elif isinstance(value, (int, float, bool)):
                scalars[key] = value
        extra = {}
        sum_tree = getattr(self, "sum_tree", None)
        if sum_tree is not None:
            extra["sum_tree_tree"] = sum_tree.tree.detach().cpu()
            extra["sum_tree_max_priority"] = sum_tree.max_priority.detach().cpu()
        T.save({"tensors": tensors, "scalars": scalars, "extra": extra}, path)

    def load_state(self, path, load_weights: bool = True) -> None:
        """Restore transitions + counters written by :meth:`save_state`."""
        ckpt = T.load(str(path), map_location=self.device, weights_only=False)
        for key, value in ckpt.get("tensors", {}).items():
            setattr(self, key, value.to(self.device))
        for key, value in ckpt.get("scalars", {}).items():
            setattr(self, key, value)
        sum_tree = getattr(self, "sum_tree", None)
        extra = ckpt.get("extra", {})
        if sum_tree is not None and "sum_tree_tree" in extra:
            sum_tree.tree = extra["sum_tree_tree"].to(self.device)
            sum_tree.max_priority = extra["sum_tree_max_priority"].to(self.device)

class ReplayBuffer(Buffer):
    """
    Off-Policy replay buffer with N-step sequence sampling.
    """
    def __init__(
        self,
        env: EnvWrapper,
        buffer_size: int = 100000,
        N: int = 1,
        hindsight: HindsightRelabeler | None = None,
        device: str | T.device | None = None,
    ):
        # If using hindsigh relabeling, check to make sure goal key is specified, output format is n_step
        # and N values matches Buffer N value
        if hindsight is not None:
            if env.goal_key is None:
                raise ValueError("ReplayBuffer requires goal key to be specified for hindsight relabeling")
            if hindsight.output_format != "n_step":
                raise ValueError("ReplayBuffer requires hindsight relabeling output_format = 'n_step'")
            if hindsight.N != N:
                raise ValueError("ReplayBuffer hindsight relabeling N value must match Buffer N value")
        
        super().__init__(env, buffer_size, hindsight, device)
        # Check to make sure environment is using VectorNStepReward Wrapper
        if env._find_nstep_wrapper() is None:
            raise ValueError("ReplayBuffer requires the VectorNStepReward wrapper to be used in the environment")
        self.N = N  # N-step hyperparameter
        self.samples_added = 0

        
        self.states = T.zeros((buffer_size, N, *self.obs_space_shape), dtype=T.float32, device=self.device)
        self.actions = T.zeros((buffer_size, N, *self.action_space_shape), dtype=self.action_type, device=self.device)
        self.raw_actions = T.zeros((buffer_size, N, *self.action_space_shape), dtype=self.action_type, device=self.device)
        self.log_probs = T.zeros((buffer_size, N), dtype=T.float32, device=self.device)
        self.rewards = T.zeros((buffer_size, N), dtype=T.float32, device=self.device)
        self.intrinsic_rewards = T.zeros((buffer_size, N), dtype=T.float32, device=self.device)
        self.next_states = T.zeros((buffer_size, N, *self.obs_space_shape), dtype=T.float32, device=self.device)
        self.terminations = T.zeros((buffer_size, N), dtype=T.bool, device=self.device)
        self.truncations = T.zeros((buffer_size, N), dtype=T.bool, device=self.device)
        self.trajectory_lengths = T.zeros((buffer_size,), dtype=T.int64, device=self.device)
        if self.env.goal_key is not None and self.goal_space_shape is not None:
            self.desired_goals = T.zeros((buffer_size, N, *self.goal_space_shape), dtype=T.float32, device=self.device)
            self.state_achieved_goals = T.zeros((buffer_size, N, *self.goal_space_shape), dtype=T.float32, device=self.device)
            self.next_state_achieved_goals = T.zeros((buffer_size, N, *self.goal_space_shape), dtype=T.float32, device=self.device)
        
        self.gen = np.random.default_rng()

    def record(
        self,
        cur_observation: Observation,
        prev_observation: Observation,
        actions: Action,
        prev_dones: T.Tensor,
        ) -> None:
        """
        Record a transition into the buffer.

        Args:
            cur_observation: Observation: The observation of the current state.
            prev_observation: Observation: The observation of the previous state.
            actions: Action: The actions taken.
            prev_dones: T.Tensor: The previous dones of the environments.
        """
        self.env_steps += self.env.num_envs
        if cur_observation.n_step_trajectory is not None:
            self.add(**cur_observation.n_step_trajectory)
        
        if self.hindsight is None:
            return

        self._her_step(cur_observation, prev_observation, actions, prev_dones)

    def _her_step(self, cur_observation: Observation, prev_observation: Observation, actions: Action, prev_dones: T.Tensor) -> None:
        """
        Adds trajectory data to the episodes buffers and adds relabeled completed trajectories to the buffer.
        """
        zero_ir = T.zeros((), dtype=T.float32, device=self.device)
        for i in range(self.env.num_envs):
            if prev_dones[i]:
                ep_buf = self._ep_buffers[i]
                if len(ep_buf["states"]) > 0:
                    episode = {k: T.stack(v) for k, v in ep_buf.items() if all(x is not None for x in v)}

                    # Add achieved goals to pool if strategy = 'random'
                    if self.hindsight.goal_pool is not None:
                        self.hindsight.goal_pool.add(episode["next_state_achieved_goals"])

                    # Relabel completed episode
                    relabeled_episode = self.hindsight.relabel_episode(episode)
                    if relabeled_episode is not None:
                        self.add(**relabeled_episode)
                    # clear episode buffer
                    self._ep_buffers[i] = self._empty_ep_buffer()
                continue
                
            # Add current step data to episode buffer
            ep_buf = self._ep_buffers[i]
            ep_buf["states"].append(prev_observation.states[i].detach().clone())
            ep_buf["actions"].append(actions.actions[i].detach().clone())
            ep_buf["rewards"].append(cur_observation.rewards[i].detach().clone())
            ir = (cur_observation.intrinsic_rewards[i].detach().clone()
              if cur_observation.intrinsic_rewards is not None else zero_ir.clone())
            ep_buf["intrinsic_rewards"].append(ir)
            ep_buf["next_states"].append(cur_observation.states[i].detach().clone())
            ep_buf["terminations"].append(cur_observation.terminations[i].detach().clone())
            ep_buf["truncations"].append(cur_observation.truncations[i].detach().clone())
            ep_buf["state_achieved_goals"].append(prev_observation.ach_goals[i].detach().clone())
            ep_buf["next_state_achieved_goals"].append(cur_observation.ach_goals[i].detach().clone())
            ep_buf["desired_goals"].append(prev_observation.goals[i].detach().clone())
            ep_buf["trajectory_lengths"].append(cur_observation.n_step_trajectory["trajectory_lengths"][i].detach().clone())

            # Add raw actions and log probs if present, else None
            raw_actions = None if actions.raw_actions is None else actions.raw_actions[i].detach().clone()
            log_probs = None if actions.log_probs is None else actions.log_probs[i].detach().clone()
            ep_buf["raw_actions"].append(raw_actions)
            ep_buf["log_probs"].append(log_probs)


    def add(
        self,
        states: T.Tensor,
        actions: T.Tensor,
        rewards: T.Tensor,
        next_states: T.Tensor,
        terminations: T.Tensor,
        truncations: T.Tensor,
        raw_actions: T.Tensor | None = None,
        log_probs: T.Tensor | None = None,
        intrinsic_rewards: T.Tensor | None = None,
        state_achieved_goals: T.Tensor | None = None,
        next_state_achieved_goals: T.Tensor | None = None,
        desired_goals: T.Tensor | None = None,
        trajectory_lengths: T.Tensor | None = None,
    ) -> None:
        batch_size = len(states)
        start_idx = self.samples_added % self.buffer_size
        end_idx = (self.samples_added + batch_size) % self.buffer_size

        if end_idx > start_idx:
            indices = T.arange(start_idx, end_idx, device=self.device)
        else:
            indices = T.cat([T.arange(start_idx, self.buffer_size, device=self.device), T.arange(0, end_idx, device=self.device)])

        # Unsqueeze to add dimenstion at index -1 if feature dim missing
        if states.ndim == 2:
            # states = states[:, T.newaxis, :]
            states = states.unsqueeze(-1)
        if actions.ndim == 2:
            # actions = actions[:, T.newaxis, :]
            actions = actions.unsqueeze(-1)
        if rewards.ndim == 1:
            # rewards = rewards[:, T.newaxis]
            rewards = rewards.unsqueeze(-1)
        if next_states.ndim == 2:
            # next_states = next_states[:, T.newaxis, :]
            next_states = next_states.unsqueeze(-1)
        if terminations.ndim == 1:
            # terminations = terminations[:, T.newaxis]
            terminations = terminations.unsqueeze(-1)
        if truncations.ndim == 1:
            # truncations = truncations[:, T.newaxis]
            truncations = truncations.unsqueeze(-1)

        if self.env.goal_key is not None and self.goal_space_shape is not None:
            if state_achieved_goals is None or next_state_achieved_goals is None or desired_goals is None:
                raise ValueError("Goal data must be provided when using goals")
            if state_achieved_goals.ndim == 2:
                # state_achieved_goals = state_achieved_goals[:, T.newaxis, :]
                state_achieved_goals = state_achieved_goals.unsqueeze(-1)
            if next_state_achieved_goals.ndim == 2:
                # next_state_achieved_goals = next_state_achieved_goals[:, T.newaxis, :]
                next_state_achieved_goals = next_state_achieved_goals.unsqueeze(-1)
            if desired_goals.ndim == 2:
                # desired_goals = desired_goals[:, T.newaxis, :]
                desired_goals = desired_goals.unsqueeze(-1)

        # Store transitions (detach to avoid holding computation graphs)
        self.states[indices] = states.detach().to(device=self.device, dtype=T.float32)
        self.actions[indices] = actions.detach().to(device=self.device, dtype=self.action_type)
        self.rewards[indices] = rewards.detach().to(device=self.device, dtype=T.float32)
        self.next_states[indices] = next_states.detach().to(device=self.device, dtype=T.float32)
        self.terminations[indices] = terminations.detach().to(device=self.device, dtype=T.bool)
        self.truncations[indices] = truncations.detach().to(device=self.device, dtype=T.bool)
        self.trajectory_lengths[indices] = trajectory_lengths.detach().to(device=self.device, dtype=T.int64)

        if raw_actions is not None:
            if raw_actions.ndim == 2:
                raw_actions = raw_actions.unsqueeze(-1)
            self.raw_actions[indices] = raw_actions.detach().to(device=self.device, dtype=self.action_type)
        if log_probs is not None:
            log_probs = log_probs.detach().to(device=self.device, dtype=T.float32)
            if log_probs.ndim == 3 and log_probs.shape[-1] == 1:
                log_probs = log_probs.squeeze(-1)
            self.log_probs[indices] = log_probs

        if intrinsic_rewards is not None:
            if intrinsic_rewards.ndim == 1:
                intrinsic_rewards = intrinsic_rewards.unsqueeze(-1)
            self.intrinsic_rewards[indices] = intrinsic_rewards.detach().to(device=self.device, dtype=T.float32)
        
        if self.env.goal_key is not None and self.goal_space_shape is not None:
            self.state_achieved_goals[indices] = state_achieved_goals.detach().to(device=self.device, dtype=T.float32)
            self.next_state_achieved_goals[indices] = next_state_achieved_goals.detach().to(device=self.device, dtype=T.float32)
            self.desired_goals[indices] = desired_goals.detach().to(device=self.device, dtype=T.float32)

        self.samples_added += batch_size

    def sample(self, samples: int) -> Dict[str, T.Tensor]:
        """Returns a dictionary of n-step sequences sampled from the buffer.
        
        Args:
            samples: int: The number of samples to draw from the buffer.
        
        Returns:
            Dict[str, T.Tensor]: A dictionary containing the sampled n-step sequences.
        """
        size = min(self.samples_added, self.buffer_size)
        if size == 0:
            raise ValueError("Cannot sample from empty buffer")

        indices = self.gen.integers(0, size, (samples,))

        sample = {
            "states": self.states[indices].clone(),
            "actions": self.actions[indices].clone(),
            "rewards": self.rewards[indices].clone(),
            "intrinsic_rewards": self.intrinsic_rewards[indices].clone(),
            "next_states": self.next_states[indices].clone(),
            "terminations": self.terminations[indices].clone(),
            "truncations": self.truncations[indices].clone(),
            "trajectory_lengths": self.trajectory_lengths[indices].clone(),
            "raw_actions": self.raw_actions[indices].clone(),
            "log_probs": self.log_probs[indices].clone(),
        }
        if self.env.goal_key is not None and self.goal_space_shape is not None:
            sample.update({
                "state_achieved_goals": self.state_achieved_goals[indices].clone(),
                "next_state_achieved_goals": self.next_state_achieved_goals[indices].clone(),
                "desired_goals": self.desired_goals[indices].clone(),
            })
        else:
            sample.update({
                "state_achieved_goals": None,
                "next_state_achieved_goals": None,
                "desired_goals": None,
            })

        return sample
    
    def reset(self) -> None:
        """
        Reset the buffer to all zeros and the counter to zero.
        """
        self.states.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.intrinsic_rewards.zero_()
        self.next_states.zero_()
        self.terminations.zero_()
        self.truncations.zero_()
        self.trajectory_lengths.zero_()
        self.raw_actions.zero_()
        self.log_probs.zero_()
        self.samples_added = 0
        self.env_steps = 0

        if self.env.goal_key is not None and self.goal_space_shape is not None:
            self.desired_goals.zero_()
            self.state_achieved_goals.zero_()
            self.next_state_achieved_goals.zero_()

    def is_ready(self, samples: int) -> bool:
        """
        Check if the buffer is ready to sample.
        """
        return self.samples_added >= samples
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config['type'] = self.__class__.__name__
        config['config'].update({
            "N": self.N
        })
        return config

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized off-policy replay buffer.
 
    Samples transitions with probability proportional to (|δ| + ε)^α and
    corrects the resulting bias with importance weights (N · P)^(-β). β is
    annealed from `beta_start` toward 1.0 over `beta_iter` gradient steps.
    """
 
    def __init__(
        self,
        env: "EnvWrapper",
        buffer_size: int = 100_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_iter: int = 100_000,
        priority: str = "proportional",
        sort_freq: int = 1000, # How often to resort the priorities (in samples added)
        epsilon: float = 1e-6,
        N: int = 1,
        hindsight: HindsightRelabeler | None = None,
        device: str | T.device | None = None,
    ):
        if priority not in ("proportional", "rank"):
            raise ValueError(
                f"Invalid priority type: {priority!r} (must be 'proportional' or 'rank')"
            )
 
        super().__init__(env, buffer_size, N, hindsight, device)
 
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_iter = beta_iter
        self.priority = priority
        self.sort_freq = sort_freq
        self.epsilon = epsilon
        self.beta = beta_start
 
        if self.priority == "proportional":
            self.sum_tree = SumTree(buffer_size, self.device)
        else:  # rank
            self.priorities = T.zeros(buffer_size, dtype=T.float32, device=self.device)
            self.sorted_indices: T.Tensor | None = None
            self.max_priority_rank = T.tensor(1.0, dtype=T.float32, device=self.device)
            self._samples_since_sort = 0
            self._seg_cache_key: Tuple[int, int] = (-1, -1)
            self._seg_starts: Optional[T.Tensor] = None
            self._seg_ends: Optional[T.Tensor] = None
            # Z = sum_r (r+1)^(-alpha) — computed alongside segments.
            self._rank_Z = T.tensor(0.0, dtype=T.float32, device=self.device)
 
    def record(
        self,
        cur_observation: Observation,
        prev_observation: Observation,
        actions: Action,
        prev_dones: T.Tensor,
        ) -> None:
        """
        Record a transition into the buffer.

        Args:
            cur_observation: Observation: The observation of the current state.
            prev_observation: Observation: The observation of the previous state.
            actions: Action: The actions taken.
            prev_dones: T.Tensor: The previous dones of the environments.
        """
        self.env_steps += self.env.num_envs
        if cur_observation.n_step_trajectory is not None:
            self.add(**cur_observation.n_step_trajectory)
        
        if self.hindsight is None:
            return
        
        self._her_step(cur_observation, prev_observation, actions, prev_dones)
 
    def add(
        self,
        states: T.Tensor,
        actions: T.Tensor,
        rewards: T.Tensor,
        next_states: T.Tensor,
        terminations: T.Tensor,
        truncations: T.Tensor,
        raw_actions: T.Tensor | None = None,
        log_probs: T.Tensor | None = None,
        intrinsic_rewards: T.Tensor | None = None,
        state_achieved_goals: T.Tensor | None = None,
        next_state_achieved_goals: T.Tensor | None = None,
        desired_goals: T.Tensor | None = None,
        trajectory_lengths: T.Tensor | None = None,
    ) -> None:

        batch_size = len(states)
        self._samples_since_sort += batch_size
        start_idx = self.samples_added % self.buffer_size
        end_idx = (self.samples_added + batch_size) % self.buffer_size
        if end_idx > start_idx:
            indices = T.arange(start_idx, end_idx, device=self.device)
        else:
            indices = T.cat([
                T.arange(start_idx, self.buffer_size, device=self.device),
                T.arange(0, end_idx, device=self.device),
            ])
 
        super().add(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            terminations=terminations,
            truncations=truncations,
            raw_actions=raw_actions,
            log_probs=log_probs,
            intrinsic_rewards=intrinsic_rewards,
            state_achieved_goals=state_achieved_goals,
            next_state_achieved_goals=next_state_achieved_goals,
            desired_goals=desired_goals,
            trajectory_lengths=trajectory_lengths,
        )
 
        if self.priority == "proportional":
            init_priorities = self.sum_tree.max_priority.expand(batch_size)
            self.sum_tree.update(indices, init_priorities)
        else:  # rank
            self.priorities[indices] = self.max_priority_rank.expand(batch_size)

        # Update beta
        self.update_beta()
 
    def sample(self, samples: int) -> Dict[str, T.Tensor]:
        """
        Sample a batch weighted by priority.

        Args:
            samples: int: The number of samples to draw from the buffer.
 
        Returns a dict with the standard transition fields plus:
            indices : buffer indices, needed for update_priorities() later
            weights : importance-sampling weights, normalized so max == 1
            probs   : sampling probabilities (priority / total)
        """
        size = min(self.samples_added, self.buffer_size)
        if size == 0:
            raise ValueError("Cannot sample from empty buffer")
        samples = min(samples, size)
 
        if self.priority == "proportional":
            indices, probs, weights = self._sample_proportional(samples, size)
        else:
            indices, probs, weights = self._sample_rank(samples, size)

        # Clamp indices to current buffer size to ensure not sampling outside bounds
        indices = indices.clamp(max=size - 1)
 
        sample = {
            "states": self.states[indices].clone(),
            "actions": self.actions[indices].clone(),
            "rewards": self.rewards[indices].clone(),
            "intrinsic_rewards": self.intrinsic_rewards[indices].clone(),
            "next_states": self.next_states[indices].clone(),
            "terminations": self.terminations[indices].clone(),
            "truncations": self.truncations[indices].clone(),
            "trajectory_lengths": self.trajectory_lengths[indices].clone(),
            "raw_actions": self.raw_actions[indices].clone(),
            "log_probs": self.log_probs[indices].clone(),
            "weights": weights,
            "probs": probs,
            "indices": indices,
        }
        if self.env.goal_key is not None and self.goal_space_shape is not None:
            sample.update({
                "state_achieved_goals": self.state_achieved_goals[indices].clone(),
                "next_state_achieved_goals": self.next_state_achieved_goals[indices].clone(),
                "desired_goals": self.desired_goals[indices].clone(),
            })
        else:
            sample.update({
                "state_achieved_goals": None,
                "next_state_achieved_goals": None,
                "desired_goals": None,
            })
        return sample
 
    def _sample_proportional(
        self, samples: int, size: int
    ) -> Tuple[T.Tensor, T.Tensor, T.Tensor]:

        total = self.sum_tree.total_priority
        if total <= 0.0:
            # Tree is empty or all-zero — fall back to uniform.
            indices = T.randint(0, size, (samples,), device=self.device)
            probs = T.full((samples,), 1.0 / size, device=self.device)
            weights = T.ones(samples, device=self.device)
            return indices, probs, weights
 
        # Stratified sampling: divide [0, total] into `samples` equal-width
        # segments, draw one cumulative-priority value uniformly from each.
        segment = total / samples
        boundaries = T.arange(samples, device=self.device, dtype=T.float32) * segment
        offsets = T.rand(samples, device=self.device) * segment
        p_values = boundaries + offsets
 
        indices, priorities = self.sum_tree.get(p_values)
        probs = priorities / total
        weights = (size * probs).pow(-self.beta)
        weights = weights / weights.max()
        return indices, probs, weights
 
    def _sample_rank(
        self, samples: int, size: int
    ) -> Tuple[T.Tensor, T.Tensor, T.Tensor]:
        """
        Stratified-segments rank-based sampling
        """
        self._maybe_resort(size)
        # pin size to sorted size to avoid OOB error
        size = min(size, self.sorted_indices.numel())
        seg_starts, seg_ends = self._get_segments(size, samples)

        seg_widths = (seg_ends - seg_starts).float()
        u = T.rand(samples, device=self.device)
        ranks = (seg_starts.float() + u * seg_widths).long().clamp(max=size - 1)

        indices = self.sorted_indices[ranks]

        probs = (ranks.float() + 1.0).pow(-self.alpha) / self._rank_Z
        weights = (size * probs).pow(-self.beta)
        weights = weights / weights.max()
        return indices, probs, weights

    def _maybe_resort(self, size: int) -> None:
        """Resort priorities[:size] if the cache is stale.

        Two trigger conditions:
          1. sorted_indices is None (first sample, or sort_freq).
          2. sorted_indices.numel() != size (buffer size changed).
        """
        needs_resort = self.sorted_indices is None

        if needs_resort:
            self.sorted_indices = T.argsort(self.priorities[:size], descending=True)
            self._samples_since_sort = 0

    def _get_segments(
        self, size: int, samples: int
    ) -> Tuple[T.Tensor, T.Tensor]:
        """Return cached (seg_starts, seg_ends), recomputing if (size, samples) changed."""
        key = (size, samples)
        if self._seg_cache_key != key:
            self._compute_segments(size, samples)
            self._seg_cache_key = key
        return self._seg_starts, self._seg_ends

    def _compute_segments(self, size: int, k: int) -> None:
        """Compute k equal-probability segment boundaries over rank distribution
        P(r) = (r+1)^(-α) / Z for r = 0..size-1.
        """
        ranks_one_indexed = T.arange(1, size + 1, dtype=T.float32, device=self.device)
        pmf = ranks_one_indexed.pow(-self.alpha)
        self._rank_Z = pmf.sum()

        cdf = T.cumsum(pmf / self._rank_Z, dim=0)

        # For each i = 1..k, find smallest rank r where cdf[r] >= i/k.
        target_cdf = T.arange(1, k + 1, dtype=T.float32, device=self.device) / k
        seg_ends_inclusive = T.searchsorted(cdf, target_cdf, right=False).clamp(max=size - 1)
        seg_ends = seg_ends_inclusive + 1  # exclusive end
        seg_starts = T.cat([
            T.zeros(1, dtype=T.long, device=self.device),
            seg_ends[:-1].clone(),
        ])
        seg_ends = T.maximum(seg_ends, seg_starts + 1).clamp(max=size)

        self._seg_starts = seg_starts
        self._seg_ends = seg_ends
 
    def update_beta(self) -> None:
        """Linearly anneal β from β_start toward 1.0 over beta_iter gradient steps."""
        progress = min(self.env_steps / self.beta_iter, 1.0)
        self.beta = self.beta_start + progress * (1.0 - self.beta_start)
 
    def update_priorities(self, indices: T.Tensor, td_errors: T.Tensor) -> None:
        """
        Recompute priorities from new TD errors for previously-sampled transitions.
 
        Stores p = (|δ| + ε)^α
        """
        if not isinstance(indices, T.Tensor):
            indices = T.tensor(indices, device=self.device, dtype=T.long)
        if not isinstance(td_errors, T.Tensor):
            td_errors = T.tensor(td_errors, device=self.device, dtype=T.float32)
 
        priorities = (T.abs(td_errors) + self.epsilon).pow(self.alpha)
 
        # Replace any NaN with the batch mean of valid entries
        if T.isnan(priorities).any():
            nan_mask = T.isnan(priorities)
            valid = priorities[~nan_mask]
            fill = (
                valid.mean()
                if valid.numel() > 0
                else T.tensor(self.epsilon ** self.alpha, device=self.device)
            )
            priorities = T.where(nan_mask, fill, priorities)
 
        if self.priority == "proportional":
            self.sum_tree.update(indices, priorities)
        else:  # rank
            self.priorities[indices] = priorities
            self.max_priority_rank = T.maximum(
                self.max_priority_rank, priorities.max()
            )
            if self._samples_since_sort >= self.sort_freq:
                self.sorted_indices = None  # invalidate sort cache
 
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["type"] = self.__class__.__name__
        config["config"].update({
            "alpha": self.alpha,
            "beta_start": self.beta_start,
            "beta_iter": self.beta_iter,
            "priority": self.priority,
            "sort_freq": self.sort_freq,
            "epsilon": self.epsilon,
        })
        return config

class RolloutBuffer(Buffer):
    """
    On-Policy buffer for storing rollouts.
    """
    def __init__(
        self,
        env: EnvWrapper,
        buffer_size: int,
        hindsight: HindsightRelabeler | None = None,
        device: str | T.device | None = None
    ):
        super().__init__(env, buffer_size, hindsight, device)
        self.cur_idx = T.zeros((env.num_envs,), dtype=T.long, device=self.device)

        #Instantiate buffers
        self.states = T.zeros((buffer_size, env.num_envs, *self.obs_space_shape), dtype=T.float32, device=self.device)
        self.actions = T.zeros((buffer_size, env.num_envs, *self.action_space_shape), dtype=self.action_type, device=self.device)
        self.raw_actions = T.zeros((buffer_size, env.num_envs, *self.action_space_shape), dtype=self.action_type, device=self.device)
        self.log_probs = T.zeros((buffer_size, env.num_envs), dtype=T.float32, device=self.device)
        self.rewards = T.zeros((buffer_size, env.num_envs), dtype=T.float32, device=self.device)
        self.intrinsic_rewards = T.zeros((buffer_size, env.num_envs), dtype=T.float32, device=self.device)
        self.next_states = T.zeros((buffer_size, env.num_envs, *self.obs_space_shape), dtype=T.float32, device=self.device)
        self.terminations = T.zeros((buffer_size, env.num_envs), dtype=T.bool, device=self.device)
        self.truncations = T.zeros((buffer_size, env.num_envs), dtype=T.bool, device=self.device)
        # Tracks initial steps of each environment (Phantom steps)
        self.first_steps = T.zeros((buffer_size, env.num_envs), dtype=T.bool, device=self.device)
        
        if self.env.goal_key is not None and self.goal_space_shape is not None:
            self.desired_goals = T.zeros((buffer_size, env.num_envs, *self.goal_space_shape), dtype=T.float32, device=self.device)
            self.state_achieved_goals = T.zeros((buffer_size, env.num_envs, *self.goal_space_shape), dtype=T.float32, device=self.device)
            self.next_state_achieved_goals = T.zeros((buffer_size, env.num_envs, *self.goal_space_shape), dtype=T.float32, device=self.device)

    def record(self, cur_observation: Observation, prev_observation: Observation, actions: Action, prev_dones: T.Tensor) -> None:
        """
        Record a transition into the buffer.

        Args:
            cur_observation: Observation: The observation of the current state.
            prev_observation: Observation: The observation of the previous state.
            actions: Action: The actions taken.
            prev_dones: T.Tensor: The previous dones of the environments.
        """
        self.add(
            states=prev_observation.states,
            actions=actions.actions,
            rewards=cur_observation.rewards,
            next_states=cur_observation.states,
            terminations=cur_observation.terminations,
            truncations=cur_observation.truncations,
            raw_actions=actions.raw_actions,
            log_probs=actions.log_probs,
            intrinsic_rewards=cur_observation.intrinsic_rewards,
            state_achieved_goals=prev_observation.ach_goals if prev_observation.ach_goals is not None else None,
            next_state_achieved_goals=cur_observation.ach_goals if cur_observation.ach_goals is not None else None,
            desired_goals=prev_observation.goals if prev_observation.goals is not None else None,
            first_steps=prev_dones,
        )

    def add(
        self,
        states: T.Tensor,
        actions: T.Tensor,
        rewards: T.Tensor,
        next_states: T.Tensor,
        terminations: T.Tensor,
        truncations: T.Tensor,
        raw_actions: T.Tensor | None = None,
        log_probs: T.Tensor | None = None,
        intrinsic_rewards: T.Tensor | None = None,
        state_achieved_goals: T.Tensor|None = None,
        next_state_achieved_goals: T.Tensor|None = None,
        desired_goals: T.Tensor|None = None,
        first_steps: T.Tensor|None = None,
    ) -> None:

        # Ensure actions are 2d tensors
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        # Check valid_steps and create if None
        if first_steps is None:
            first_steps = T.zeros((self.env.num_envs,), dtype=T.bool, device=self.device)
        else:
            first_steps = first_steps.to(device=self.device, dtype=T.bool)
            if first_steps.numel() != self.env.num_envs:
                raise ValueError(f"first_steps must have {self.env.num_envs} elements, got {first_steps.numel()}")
        # Get per environment indices to add values to
        idx = self.cur_idx.clone()
        # Get env id's
        env_ids = T.arange(self.env.num_envs, device=self.device)

        
        # Add values to buffers at indices
        self.states[idx, env_ids] = states.to(device=self.device, dtype=T.float32)
        self.actions[idx, env_ids] = actions.to(device=self.device, dtype=self.action_type)
        self.rewards[idx, env_ids] = rewards.to(device=self.device, dtype=T.float32)
        if intrinsic_rewards is not None:
            self.intrinsic_rewards[idx, env_ids] = intrinsic_rewards.to(device=self.device, dtype=T.float32)
        self.next_states[idx, env_ids] = next_states.to(device=self.device, dtype=T.float32)
        self.terminations[idx, env_ids] = terminations.to(device=self.device, dtype=T.bool)
        self.truncations[idx, env_ids] = truncations.to(device=self.device, dtype=T.bool)
        self.first_steps[idx, env_ids] = first_steps

        if raw_actions is not None:
            self.raw_actions[idx, env_ids] = raw_actions.to(device=self.device, dtype=self.action_type)
        if log_probs is not None:
            self.log_probs[idx, env_ids] = log_probs.to(device=self.device, dtype=T.float32)

        if self.env.goal_key is not None and self.goal_space_shape is not None:
            self.state_achieved_goals[idx, env_ids] = state_achieved_goals.to(device=self.device, dtype=T.float32)
            self.next_state_achieved_goals[idx, env_ids] = next_state_achieved_goals.to(device=self.device, dtype=T.float32)
            self.desired_goals[idx, env_ids] = desired_goals.to(device=self.device, dtype=T.float32)
        
        # Increment step indices
        self.cur_idx += 1

    def sample(self, **kwargs: Any) -> Dict[str, T.Tensor]:
        """
        Returns a dictionary of all buffer tensors up to the current index of each environment.
        Current index values will match across all tensors because all rollouts are of same length.
        """
        idx = int(self.cur_idx.max().item())
        if idx <= 0:
            raise ValueError("Cannot sample from empty buffer")

        # Create tensor of non first step (Phantom) indices for valid training samples
        first_steps = self.first_steps[:idx].clone()
        valid_indices = (first_steps.reshape(-1) == 0).nonzero()
        sample = {
            "states": self.states[:idx].clone(),
            "actions": self.actions[:idx].clone(),
            "rewards": self.rewards[:idx].clone(),
            "intrinsic_rewards": self.intrinsic_rewards[:idx].clone(),
            "next_states": self.next_states[:idx].clone(),
            "terminations": self.terminations[:idx].clone(),
            "truncations": self.truncations[:idx].clone(),
            "raw_actions": self.raw_actions[:idx].clone(),
            "log_probs": self.log_probs[:idx].clone(),
            "first_steps": first_steps,
            "valid_indices": valid_indices,
        }
        if self.env.goal_key is not None and self.goal_space_shape is not None:
            sample.update({
                "state_achieved_goals": self.state_achieved_goals[:idx].clone(),
                "next_state_achieved_goals": self.next_state_achieved_goals[:idx].clone(),
                "desired_goals": self.desired_goals[:idx].clone(),
            })
        else:
            sample.update({
                "state_achieved_goals": None,
                "next_state_achieved_goals": None,
                "desired_goals": None,
            })

        self.reset()
        return sample

    def reset(self) -> None:
        """Reset the current index of each environment to zero."""
        self.cur_idx.zero_()
        self.first_steps.zero_()

    def is_ready(self, **kwargs: Any) -> bool:
        """
        Check if the buffer is ready to sample. Always returns True.
        """
        return True
    
    def get_config(self) -> Dict[str, Any]:
        """Get buffer config."""
        config = super().get_config()
        config['type'] = self.__class__.__name__
        return config

class TrajectoryBuffer(RolloutBuffer):
    """
    On-Policy buffer for storing completedtrajectories
    """
    def __init__(
        self,
        env: EnvWrapper,
        buffer_size: int,
        hindsight: HindsightRelabeler | None = None,
        device: Optional[str] = None
    ):
        # Check hindisght if None to make sure correct
        if hindsight is not None:
            if env.goal_key is None:
                raise ValueError("TrajectoryBuffer requires goal key to be specified for hindsight relabeling")
            if hindsight.output_format != "flat":
                raise ValueError("TrajectoryBuffer requires hindsight relabeling output_format = 'flat',"
                                 f"got {hindsight.output_format}")

        super().__init__(env, buffer_size, hindsight, device)
        self.completed_trajectories: List[Dict[str, T.Tensor]] = []


    def record(self, cur_observation: Observation, prev_observation: Observation, actions: Action, prev_dones: T.Tensor) -> None:
        """
        Record a transition into the buffer.

        Args:
            cur_observation: Observation: The observation of the current state.
            prev_observation: Observation: The observation of the previous state.
            actions: Action: The actions taken.
            prev_dones: T.Tensor: The previous dones of the environments.
        """
        self.add(
            states=prev_observation.states,
            actions=actions.actions,
            rewards=cur_observation.rewards,
            next_states=cur_observation.states,
            terminations=cur_observation.terminations,
            truncations=cur_observation.truncations,
            raw_actions=actions.raw_actions,
            log_probs=actions.log_probs,
            intrinsic_rewards=cur_observation.intrinsic_rewards,
            state_achieved_goals=prev_observation.ach_goals if prev_observation.ach_goals is not None else None,
            next_state_achieved_goals=cur_observation.ach_goals if cur_observation.ach_goals is not None else None,
            desired_goals=prev_observation.goals if prev_observation.goals is not None else None,
            first_steps=prev_dones,
        )

    def add(
        self,
        states: T.Tensor,
        actions: T.Tensor,
        rewards: T.Tensor,
        next_states: T.Tensor,
        terminations: T.Tensor,
        truncations: T.Tensor,
        raw_actions: T.Tensor | None = None,
        log_probs: T.Tensor | None = None,
        intrinsic_rewards: T.Tensor | None = None,
        state_achieved_goals: T.Tensor|None = None,
        next_state_achieved_goals: T.Tensor|None = None,
        desired_goals: T.Tensor|None = None,
        first_steps: T.Tensor|None = None,
    ) -> None:
        super().add(
            states,
            actions,
            rewards,
            next_states,
            terminations,
            truncations,
            raw_actions,
            log_probs,
            intrinsic_rewards,
            state_achieved_goals,
            next_state_achieved_goals,
            desired_goals,
            first_steps
        )

        # Store completed trajectories if any last stored dones are True
        for i in range(self.env.num_envs):
            idx = int(self.cur_idx[i].item())
            if idx <= 0:
                continue
            done = bool(self.terminations[idx - 1, i].item() or self.truncations[idx - 1, i].item())
            if not done:
                continue
            # Check to make sure there are valid steps in the trajectory
            valid_steps = T.logical_not(self.first_steps[:idx, i])
            if not valid_steps.any():
                self.cur_idx[i] = 0
                continue

            trajectory = {
                "states": self.states[:idx, i][valid_steps].clone(),
                "actions": self.actions[:idx, i][valid_steps].clone(),
                "rewards": self.rewards[:idx, i][valid_steps].clone(),
                "intrinsic_rewards": self.intrinsic_rewards[:idx, i][valid_steps].clone(),
                "next_states": self.next_states[:idx, i][valid_steps].clone(),
                "terminations": self.terminations[:idx, i][valid_steps].clone(),
                "truncations": self.truncations[:idx, i][valid_steps].clone(),
                "raw_actions": self.raw_actions[:idx, i][valid_steps].clone(),
                "log_probs": self.log_probs[:idx, i][valid_steps].clone(),
            }
            if self.env.goal_key is not None and self.goal_space_shape is not None:
                trajectory.update({
                    "state_achieved_goals": self.state_achieved_goals[:idx, i][valid_steps].clone(),
                    "next_state_achieved_goals": self.next_state_achieved_goals[:idx, i][valid_steps].clone(),
                    "desired_goals": self.desired_goals[:idx, i][valid_steps].clone(),
                })
            else:
                trajectory.update({
                    "state_achieved_goals": None,
                    "next_state_achieved_goals": None,
                    "desired_goals": None,
                })
            self.completed_trajectories.append(trajectory)
            # Reset step counter for done env
            self.cur_idx[i] = 0

            # Relabel trajectory if using hindsight
            if self.hindsight is not None:
                if self.hindsight.goal_pool is not None:
                    self.hindsight.goal_pool.add(trajectory["next_state_achieved_goals"])
                self.completed_trajectories.extend(self.hindsight.relabel_episode(trajectory))

    def sample(self, **kwargs: Any) -> List[Dict[str, T.Tensor]]:
        """Returns a list of completed trajectories."""
        trajectories = self.completed_trajectories[:]
        # Clear trajectories
        self.completed_trajectories = []
        return trajectories

    def reset(self) -> None:
        super().reset()
        self.completed_trajectories = []