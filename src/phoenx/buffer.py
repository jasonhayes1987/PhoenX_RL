"""Experience storage for off-policy and on-policy training.

Off-policy paths use ``ReplayBuffer`` / ``PrioritizedReplayBuffer`` (circular
N-step windows, optional R2D2 stored state and HER). On-policy paths use
``RolloutBuffer`` / ``TrajectoryBuffer`` (fixed-horizon rollouts or
completed-episode lists). ``SumTree`` backs proportional prioritized sampling.
"""

from typing import Optional, Tuple, List, Any, Dict
from abc import ABC, abstractmethod
import torch as T
import numpy as np
import gymnasium as gym

from .env_wrapper import Observation, Action, EnvWrapper
from .her import HindsightRelabeler
from .obs_utils import (
    alloc_from_spec, obs_batch_size, obs_spec_from_space, tree_assign,
    tree_clone, tree_detach_clone, tree_index, tree_map, tree_stack, tree_zero_,
)
from .torch_utils import get_device


class SumTree:
    """Binary sum tree for priority-based sampling."""

    def __init__(self, capacity: int, device: T.device):
        """Allocate a flat heap array sized to the next power of two.

        Args:
            capacity: Requested leaf capacity; rounded up to a power of two
                for the tree layout.
            device: Device that stores the tree tensor.
        """
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
        """Write priorities at data indices and propagate sums up to root.

        Priorities are clamped to a floor of ``1e-6``. ``max_priority`` tracks
        the largest value seen.

        Args:
            data_indices: Leaf / buffer indices to update.
            priorities: Priority values written at those leaves.
        """
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
        """Map cumulative-priority values to data indices and leaf priorities.

        Args:
            p_values: Cumulative-priority probes in ``[0, total_priority]``.

        Returns:
            data_indices: Buffer indices corresponding to the selected leaves.
            priorities: Stored priorities at those leaves.
        """
        tree_indices = self._traverse(p_values)
        # Defensive clamp — shouldn't be needed if the tree is well-formed.
        tree_indices = T.clamp(tree_indices, 0, self.tree.size(0) - 1)
        priorities = self.tree[tree_indices]
        data_indices = (tree_indices - (self.capacity - 1)).clamp(0, self.capacity - 1)
        return data_indices, priorities
 
    def _traverse(self, p_values: T.Tensor) -> T.Tensor:
        """Vectorized batch descent from root to leaf.

        At each level every batch element is at some node ``idx``; read its
        left child's stored sum and decide left vs. right based on whether
        ``p`` fits in the left subtree.

        Args:
            p_values: Cumulative-priority probes (mutated via a local clone).

        Returns:
            Tree node indices of the selected leaves.
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
        """Sum of all leaf priorities (stored at the root)."""
        return self.tree[0].item()

    def reset(self) -> None:
        """Zero all stored priorities and restore ``max_priority`` to ``1.0``.

        Called by ``PrioritizedReplayBuffer.reset`` so a cleared buffer does
        not keep sampling stale leaves from the previous contents.
        """
        self.tree.zero_()
        self.max_priority = T.tensor(1.0, dtype=T.float32, device=self.device)

class Buffer(ABC):
    """Abstract base for experience buffers with optional HER episode bookkeeping."""

    def __init__(
        self,
        env: EnvWrapper,
        buffer_size: int,
        hindsight: HindsightRelabeler | None = None,
        device: Optional[str] = None
    ):
        """Store env specs, optional HER episode buffers, and device.

        Args:
            env: Wrapped environment whose observation / action spaces size
                the storage tensors.
            buffer_size: Capacity in stored transitions or windows (subclass-
                dependent layout).
            hindsight: Optional HER relabeler; when set, allocates per-env
                episode accumulators for ``_her_step``.
            device: Torch device string; ``None`` resolves via ``get_device``.
        """
        self.env = env
        self.buffer_size = buffer_size
        self.hindsight = hindsight
        if self.hindsight is not None:
            # Initialize episode trajectory buffers for each environment to
            # rebuild the trajectories with hindsight relabeling.
            self._ep_buffers = [self._empty_ep_buffer() for _ in range(env.num_envs)]
        self.device = get_device(device)
        self.env_steps = 0 # Tracks how many steps have been taken in the environment

        # Set observation, goal, and action space storage specs.
        # obs_spec is either a single (shape, dtype) tuple (flat obs) or a
        # {key: (shape, dtype)} dict (multi-modal Dict obs with obs_key=None).
        obs_space = self.env.single_observation_space
        self.obs_spec = obs_spec_from_space(
            obs_space, self.env.obs_key, (self.env.goal_key, self.env.ach_goal_key)
        )
        if isinstance(obs_space, gym.spaces.Dict):
            if self.env.obs_key is not None:
                self.obs_space_shape = obs_space[self.env.obs_key].shape
            else:
                self.obs_space_shape = None  # multi-modal: shapes live in obs_spec
            if self.env.goal_key is not None:
                self.goal_space_shape = obs_space[self.env.goal_key].shape
            else:
                self.goal_space_shape = None
        else:
            self.obs_space_shape = obs_space.shape
            self.goal_space_shape = None

        if isinstance(self.env.single_action_space, gym.spaces.Box):
            self.action_type = T.float32
            self.action_space_shape = self.env.single_action_space.shape
        else: # Discrete
            self.action_type = T.long
            self.action_space_shape = (1,)

        # Storage dtypes are reconciled against the FIRST real observation:
        # some envs misreport dtypes in their observation space (IsaacLab
        # manager-based envs declare float32 Boxes even for uint8 camera
        # groups). Trusting the space would silently store uint8 frames as
        # float 0..255 and bypass the models' uint8 -> [0,1] input scaling.
        self._storage_dtypes_synced = False

    def _sync_storage_dtypes(self, states) -> None:
        """One-time cast of per-key state storage to the actual data dtypes.

        Only applies to multi-modal dict storage (flat observations keep the
        legacy float32 storage). Casting (rather than reallocating) preserves
        any content restored by ``load_state``.

        Args:
            states (Any): Incoming observation batch (tensor or dict of
                tensors) whose dtypes are trusted over the space metadata.
        """
        if self._storage_dtypes_synced:
            return
        self._storage_dtypes_synced = True
        if not (isinstance(self.obs_spec, dict) and isinstance(states, dict)):
            return
        changed = {
            k: (shape, states[k].dtype)
            for k, (shape, dtype) in self.obs_spec.items()
            if k in states and T.is_tensor(states[k]) and states[k].dtype != dtype
        }
        if not changed:
            return
        self.obs_spec = {**self.obs_spec, **changed}
        for k, (_, dtype) in changed.items():
            self.states[k] = self.states[k].to(dtype)
            self.next_states[k] = self.next_states[k].to(dtype)

    def _empty_ep_buffer(self) -> Dict[str, List[T.Tensor]]:
        return {
            "states": [], "actions": [], "raw_actions": [], "log_probs": [],
            "rewards": [], "intrinsic_rewards": [],
            "next_states": [], "terminations": [], "truncations": [],
            "state_achieved_goals": [], "next_state_achieved_goals": [],
            "desired_goals": [], "trajectory_lengths": [],
        }

    @abstractmethod
    def add(self, *args: Any, **kwargs: Any) -> None:
        """Add a transition (or window) to the buffer.

        Abstract; concrete subclasses define the real signature:
        ``ReplayBuffer.add`` / ``PrioritizedReplayBuffer.add`` take
        ``(states, actions, rewards, next_states, terminations, truncations,
        raw_actions=None, log_probs=None, intrinsic_rewards=None,
        state_achieved_goals=None, next_state_achieved_goals=None,
        desired_goals=None, trajectory_lengths=None, initial_hidden=None)``;
        ``RolloutBuffer.add`` / ``TrajectoryBuffer.add`` take the same
        transition fields through ``desired_goals``, then ``first_steps`` in
        place of ``trajectory_lengths`` and ``initial_hidden``.

        Args:
            *args (Any): Positional transition fields; see the concrete
                subclass signature.
            **kwargs (Any): Keyword transition fields; see the concrete
                subclass signature.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> Any:
        """Sample a batch of transitions from the buffer.

        Abstract; concrete subclasses define the real signature:
        ``ReplayBuffer.sample`` / ``PrioritizedReplayBuffer.sample`` take
        ``(samples)``; ``RolloutBuffer.sample`` / ``TrajectoryBuffer.sample``
        take ``(**kwargs)`` (ignored) and return everything stored so far.

        Args:
            *args (Any): Positional arguments; see the concrete subclass
                signature.
            **kwargs (Any): Keyword arguments; see the concrete subclass
                signature.

        Returns:
            Subclass-defined batch structure (a dict, or a list of
                trajectory dicts).
        """
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        """Return a serializable ``{'type', 'config'}`` mapping for this buffer.

        Returns:
            Mapping with ``type`` set to the class name and ``config`` holding
                ``env``, ``buffer_size``, ``hindsight``, and ``device``.
        """
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
        """Construct a concrete buffer by class-name string.

        Args:
            buffer_type: One of ``"ReplayBuffer"``,
                ``"PrioritizedReplayBuffer"``, ``"RolloutBuffer"``, or
                ``"TrajectoryBuffer"``.
            **kwargs (Any): Forwarded to the concrete constructor.

        Returns:
            New buffer instance of the requested type.

        Raises:
            ValueError: If ``buffer_type`` is not a known subclass name.
        """
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
        """Persist stored transitions and counters for optional resume.

        Generic over subclasses: every tensor attribute (including dict-of-
        tensor storage for multi-modal observations) and every int/float/bool
        counter is dumped. The ``SumTree`` of a prioritized buffer is handled
        explicitly since it is a nested object rather than a bare tensor.

        Args:
            path (str | Path): Destination file path; parent dirs are created.
        """
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors, scalars = {}, {}
        for key, value in self.__dict__.items():
            if isinstance(value, T.Tensor):
                tensors[key] = value.detach().cpu()
            elif (isinstance(value, dict) and value
                  and all(isinstance(v, T.Tensor) for v in value.values())):
                tensors[key] = {k: v.detach().cpu() for k, v in value.items()}
            elif isinstance(value, (int, float, bool)):
                scalars[key] = value
        extra = {}
        sum_tree = getattr(self, "sum_tree", None)
        if sum_tree is not None:
            extra["sum_tree_tree"] = sum_tree.tree.detach().cpu()
            extra["sum_tree_max_priority"] = sum_tree.max_priority.detach().cpu()
        T.save({"tensors": tensors, "scalars": scalars, "extra": extra}, path)

    def load_state(self, path, load_weights: bool = True) -> None:
        """Restore transitions and counters written by ``save_state``.

        Args:
            path (str | Path): Checkpoint path previously written by
                ``save_state``.
            load_weights: Accepted for API symmetry with agents; ignored
                (buffers have no separate weight tensors beyond storage).
        """
        ckpt = T.load(str(path), map_location=self.device, weights_only=False)
        for key, value in ckpt.get("tensors", {}).items():
            if isinstance(value, dict):
                setattr(self, key, {k: v.to(self.device) for k, v in value.items()})
            else:
                setattr(self, key, value.to(self.device))
        for key, value in ckpt.get("scalars", {}).items():
            setattr(self, key, value)
        sum_tree = getattr(self, "sum_tree", None)
        extra = ckpt.get("extra", {})
        if sum_tree is not None and "sum_tree_tree" in extra:
            sum_tree.tree = extra["sum_tree_tree"].to(self.device)
            sum_tree.max_priority = extra["sum_tree_max_priority"].to(self.device)

class ReplayBuffer(Buffer):
    """Off-policy circular buffer of N-step transition windows."""

    def __init__(
        self,
        env: EnvWrapper,
        buffer_size: int = 100000,
        N: int = 1,
        hindsight: HindsightRelabeler | None = None,
        device: str | T.device | None = None,
    ):
        """Allocate ``(buffer_size, N, ...)`` storage for off-policy windows.

        Requires ``VectorNStepReward`` on ``env``. When ``hindsight`` is set,
        it must use ``output_format='n_step'`` with matching ``N``.

        Args:
            env: Wrapped env; must expose n-step trajectory windows.
            buffer_size: Maximum number of N-step windows retained.
            N: Window length stored along the second axis.
            hindsight: Optional HER relabeler (``n_step`` output only).
            device: Torch device; ``None`` resolves via ``get_device``.
        """
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
            raise ValueError(
                f"{type(self).__name__} requires the VectorNStepReward wrapper on the "
                "environment. Add {\"type\": \"VectorNStepReward\", \"params\": {\"n\": 1}} "
                "(n >= 1) to the environment config's wrappers list. Off-policy buffers "
                "store the (B, N) n-step trajectory windows this wrapper emits."
            )
        self.N = N  # N-step hyperparameter
        self.samples_added = 0

        # states/next_states allocate from the obs spec: a single tensor for
        # flat observations or a dict-of-tensors for multi-modal Dict obs
        # (per-key dtype preserved — uint8 images stay uint8 in storage).
        self.states = alloc_from_spec(self.obs_spec, (buffer_size, N), self.device)
        self.actions = T.zeros((buffer_size, N, *self.action_space_shape), dtype=self.action_type, device=self.device)
        self.raw_actions = T.zeros((buffer_size, N, *self.action_space_shape), dtype=self.action_type, device=self.device)
        self.log_probs = T.zeros((buffer_size, N), dtype=T.float32, device=self.device)
        self.rewards = T.zeros((buffer_size, N), dtype=T.float32, device=self.device)
        self.intrinsic_rewards = T.zeros((buffer_size, N), dtype=T.float32, device=self.device)
        self.next_states = alloc_from_spec(self.obs_spec, (buffer_size, N), self.device)
        self.terminations = T.zeros((buffer_size, N), dtype=T.bool, device=self.device)
        self.truncations = T.zeros((buffer_size, N), dtype=T.bool, device=self.device)
        self.trajectory_lengths = T.zeros((buffer_size,), dtype=T.int64, device=self.device)
        # R2D2 stored state: recurrent hidden at each window's first step,
        # flattened per key (lazily allocated on the first add carrying it).
        self.initial_hidden: Dict[str, T.Tensor] | None = None
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
        """Record one env step: store any ready n-step window, then HER bookkeeping.

        When ``cur_observation.n_step_trajectory`` is set, unpacks it into
        ``add``. If HER is enabled, delegates to ``_her_step``.

        Args:
            cur_observation: Current-step observation (may carry an n-step
                trajectory dict from ``VectorNStepReward``).
            prev_observation: Observation from the previous step.
            actions: Actions taken from ``prev_observation``.
            prev_dones: Per-env done flags from the previous step (episode
                boundaries for HER accumulation).
        """
        self.env_steps += self.env.num_envs
        if cur_observation.n_step_trajectory is not None:
            self.add(**cur_observation.n_step_trajectory)
        
        if self.hindsight is None:
            return

        self._her_step(cur_observation, prev_observation, actions, prev_dones)

    def _her_step(self, cur_observation: Observation, prev_observation: Observation, actions: Action, prev_dones: T.Tensor) -> None:
        """Accumulate per-env episode steps and add HER-relabeled windows on done.

        On ``prev_dones[i]``, stacks the env's episode buffer, optionally
        feeds the goal pool, calls ``hindsight.relabel_episode``, and
        ``add``s any relabeled n-step batch. Otherwise appends the current
        step into that env's episode buffer.

        Args:
            cur_observation: Current-step observation and rewards.
            prev_observation: Previous-step observation (states / goals).
            actions: Actions taken from ``prev_observation``.
            prev_dones: Per-env done flags; ``True`` closes that env's episode.
        """
        zero_ir = T.zeros((), dtype=T.float32, device=self.device)
        for i in range(self.env.num_envs):
            if prev_dones[i]:
                ep_buf = self._ep_buffers[i]
                if len(ep_buf["states"]) > 0:
                    # tree_stack handles both tensor and dict-of-tensor steps
                    episode = {
                        k: tree_stack(v)
                        for k, v in ep_buf.items() if all(x is not None for x in v)
                    }

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
            ep_buf["states"].append(tree_detach_clone(tree_index(prev_observation.states, i)))
            ep_buf["actions"].append(actions.actions[i].detach().clone())
            ep_buf["rewards"].append(cur_observation.rewards[i].detach().clone())
            ir = (cur_observation.intrinsic_rewards[i].detach().clone()
              if cur_observation.intrinsic_rewards is not None else zero_ir.clone())
            ep_buf["intrinsic_rewards"].append(ir)
            ep_buf["next_states"].append(tree_detach_clone(tree_index(cur_observation.states, i)))
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
        initial_hidden: Dict[str, T.Tensor] | None = None,
    ) -> None:
        """Write a batch of N-step windows into the circular store.

        Indices wrap with ``samples_added % buffer_size``. Optional
        ``initial_hidden`` allocates R2D2 storage on first use; later adds
        without it zero that slot. Detaches tensors before writing.

        Args:
            states: Window states ``(B, N, ...)`` (or dict of tensors).
            actions: Window actions ``(B, N, ...)``.
            rewards: Window rewards ``(B, N)``.
            next_states: Window next-states ``(B, N, ...)``.
            terminations: Window termination flags ``(B, N)``.
            truncations: Window truncation flags ``(B, N)``.
            raw_actions: Optional pre-squash / raw actions.
            log_probs: Optional action log-probabilities ``(B, N)``.
            intrinsic_rewards: Optional intrinsic rewards ``(B, N)``.
            state_achieved_goals: Required when the env has a goal key.
            next_state_achieved_goals: Required when the env has a goal key.
            desired_goals: Required when the env has a goal key.
            trajectory_lengths: Valid length of each window (partial at
                episode ends), shape ``(B,)``.
            initial_hidden: Optional R2D2 recurrent state at each window's
                first step (dict of tensors with leading batch dim ``B``).
        """
        batch_size = obs_batch_size(states)
        start_idx = self.samples_added % self.buffer_size
        end_idx = (self.samples_added + batch_size) % self.buffer_size

        if end_idx > start_idx:
            indices = T.arange(start_idx, end_idx, device=self.device)
        else:
            indices = T.cat([T.arange(start_idx, self.buffer_size, device=self.device), T.arange(0, end_idx, device=self.device)])

        # R2D2 stored state: lazily allocate on first sight, then ring-write.
        if initial_hidden is not None:
            if self.initial_hidden is None:
                self.initial_hidden = {
                    k: T.zeros((self.buffer_size, *v.shape[1:]), dtype=T.float32, device=self.device)
                    for k, v in initial_hidden.items()
                }
            tree_assign(self.initial_hidden, indices, initial_hidden)
        elif self.initial_hidden is not None:
            # Entries without stored state (e.g. HER-relabeled windows) fall
            # back to zero-initialized hidden.
            for buf in self.initial_hidden.values():
                buf[indices] = 0.0

        # Unsqueeze to add dimenstion at index -1 if feature dim missing
        # (per-leaf for dict observations: scalar-feature leaves only)
        states = tree_map(lambda x: x.unsqueeze(-1) if x.ndim == 2 else x, states)
        if actions.ndim == 2:
            # actions = actions[:, T.newaxis, :]
            actions = actions.unsqueeze(-1)
        if rewards.ndim == 1:
            # rewards = rewards[:, T.newaxis]
            rewards = rewards.unsqueeze(-1)
        next_states = tree_map(lambda x: x.unsqueeze(-1) if x.ndim == 2 else x, next_states)
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

        # Store transitions (detach to avoid holding computation graphs);
        # tree_assign casts each leaf to its storage dtype/device.
        self._sync_storage_dtypes(states)
        tree_assign(self.states, indices, tree_map(lambda x: x.detach(), states))
        self.actions[indices] = actions.detach().to(device=self.device, dtype=self.action_type)
        self.rewards[indices] = rewards.detach().to(device=self.device, dtype=T.float32)
        tree_assign(self.next_states, indices, tree_map(lambda x: x.detach(), next_states))
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
        """Uniformly sample n-step windows and return cloned tensors.

        Draws indices from ``[0, min(samples_added, buffer_size))``. Does not
        mutate storage. Goal keys are ``None`` when the env has no goal space.

        Args:
            samples: Number of windows to draw.

        Returns:
            Dict of cloned batch tensors (states, actions, rewards, …),
                optionally including ``initial_hidden`` and goal fields.
        """
        size = min(self.samples_added, self.buffer_size)
        if size == 0:
            raise ValueError("Cannot sample from empty buffer")

        indices = self.gen.integers(0, size, (samples,))

        sample = {
            "states": tree_clone(tree_index(self.states, indices)),
            "actions": self.actions[indices].clone(),
            "rewards": self.rewards[indices].clone(),
            "intrinsic_rewards": self.intrinsic_rewards[indices].clone(),
            "next_states": tree_clone(tree_index(self.next_states, indices)),
            "terminations": self.terminations[indices].clone(),
            "truncations": self.truncations[indices].clone(),
            "trajectory_lengths": self.trajectory_lengths[indices].clone(),
            "raw_actions": self.raw_actions[indices].clone(),
            "log_probs": self.log_probs[indices].clone(),
        }
        if self.initial_hidden is not None:
            sample["initial_hidden"] = tree_clone(tree_index(self.initial_hidden, indices))
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
        """Zero transition storage and counters; leave ``initial_hidden`` untouched.

        Stale R2D2 state is unreachable: ``samples_added`` resets to 0 so
        ``sample`` only indexes freshly written slots, and ``add`` zeroes
        ``initial_hidden`` entries written without stored state.
        """
        tree_zero_(self.states)
        self.actions.zero_()
        self.rewards.zero_()
        self.intrinsic_rewards.zero_()
        tree_zero_(self.next_states)
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
        """Return whether at least ``samples`` windows have been added.

        Args:
            samples: Minimum ``samples_added`` required before learning.
        """
        return self.samples_added >= samples
    
    def get_config(self) -> Dict[str, Any]:
        """Extend the base config with the N-step window length.

        Returns:
            Base ``get_config`` mapping with ``config['N']`` set.
        """
        config = super().get_config()
        config['type'] = self.__class__.__name__
        config['config'].update({
            "N": self.N
        })
        return config

class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized off-policy replay buffer.
 
    Samples transitions with probability proportional to (|δ| + ε)^α and
    corrects the resulting bias with importance weights (N · P)^(-β). β is
    annealed from ``beta_start`` toward 1.0 over ``update_beta`` calls.
    """
 
    def __init__(
        self,
        env: "EnvWrapper",
        buffer_size: int = 100_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_iter: int = 100_000,
        beta_update_freq: int = 1, # Anneal beta every this many sample() calls
        priority: str = "proportional",
        sort_freq: int = 1000, # How often to resort the priorities (in samples added)
        epsilon: float = 1e-6,
        N: int = 1,
        hindsight: HindsightRelabeler | None = None,
        device: str | T.device | None = None,
    ):
        """Configure priority mode, IS-weight beta schedule, and backing store.

        Args:
            env: Wrapped env (same constraints as ``ReplayBuffer``).
            buffer_size: Maximum number of N-step windows retained.
            alpha: Exponent on ``(|td_error| + epsilon)`` for priorities;
                also the rank-PMF exponent when ``priority='rank'``.
            beta_start: Initial importance-sampling exponent β.
            beta_iter: Number of ``update_beta`` calls to reach β = 1.0.
            beta_update_freq: Call ``update_beta`` every this many
                ``sample`` invocations.
            priority: ``"proportional"`` (sum-tree) or ``"rank"``.
            sort_freq: For rank mode, force a resort of the sorted-index
                cache once this many samples have been added since the last
                sort (checked on each ``sample`` / ``_maybe_resort`` call).
            epsilon: Floor added inside ``(|δ| + ε)^α``.
            N: N-step window length (forwarded to ``ReplayBuffer``).
            hindsight: Optional HER relabeler.
            device: Torch device; ``None`` resolves via ``get_device``.
        """
        if priority not in ("proportional", "rank"):
            raise ValueError(
                f"Invalid priority type: {priority!r} (must be 'proportional' or 'rank')"
            )
 
        super().__init__(env, buffer_size, N, hindsight, device)
 
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_iter = beta_iter
        self.beta_update_freq = beta_update_freq
        self.priority = priority
        self.sort_freq = sort_freq
        self.epsilon = epsilon
        self.beta = beta_start
        # β annealing progresses per sample() call (i.e. per gradient step),
        # gated by beta_update_freq.
        self._sample_calls = 0
        self._beta_updates = 0
        # Samples added since the last rank resort (tracked for both priority
        # modes; only the rank strategy consumes it).
        self._samples_since_sort = 0
 
        if self.priority == "proportional":
            self.sum_tree = SumTree(buffer_size, self.device)
        else:  # rank
            self.priorities = T.zeros(buffer_size, dtype=T.float32, device=self.device)
            self.sorted_indices: T.Tensor | None = None
            self.max_priority_rank = T.tensor(1.0, dtype=T.float32, device=self.device)
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
        """Record one env step via ``add`` (with priorities) and optional HER.

        Same flow as ``ReplayBuffer.record``, but ``add`` also seeds priorities.

        Args:
            cur_observation: Current-step observation (may carry an n-step
                trajectory dict).
            prev_observation: Observation from the previous step.
            actions: Actions taken from ``prev_observation``.
            prev_dones: Per-env done flags from the previous step.
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
        initial_hidden: Dict[str, T.Tensor] | None = None,
    ) -> None:
        """Store windows via ``ReplayBuffer.add``, then seed max priority.

        New entries receive the current max priority (sum-tree or rank array)
        so they are sampled until ``update_priorities`` revises them.

        Args:
            states: Window states ``(B, N, ...)``.
            actions: Window actions ``(B, N, ...)``.
            rewards: Window rewards ``(B, N)``.
            next_states: Window next-states ``(B, N, ...)``.
            terminations: Window termination flags ``(B, N)``.
            truncations: Window truncation flags ``(B, N)``.
            raw_actions: Optional pre-squash / raw actions.
            log_probs: Optional action log-probabilities.
            intrinsic_rewards: Optional intrinsic rewards.
            state_achieved_goals: Required when the env has a goal key.
            next_state_achieved_goals: Required when the env has a goal key.
            desired_goals: Required when the env has a goal key.
            trajectory_lengths: Valid length of each window, shape ``(B,)``.
            initial_hidden: Optional R2D2 recurrent state at window starts.
        """
        batch_size = obs_batch_size(states)
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
            initial_hidden=initial_hidden,
        )
 
        if self.priority == "proportional":
            init_priorities = self.sum_tree.max_priority.expand(batch_size)
            self.sum_tree.update(indices, init_priorities)
        else:  # rank
            self.priorities[indices] = self.max_priority_rank.expand(batch_size)
 
    def sample(self, samples: int) -> Dict[str, T.Tensor]:
        """Sample a batch weighted by priority and anneal β when due.

        Every ``beta_update_freq``-th call runs ``update_beta``. Returns the
        usual transition fields plus ``indices``, ``weights`` (IS weights
        normalized so max == 1), and ``probs``.

        Args:
            samples: Requested batch size (capped at current buffer size).

        Returns:
            Dict of cloned batch tensors including ``indices``, ``weights``,
                and ``probs`` for the subsequent ``update_priorities`` call.
        """
        size = min(self.samples_added, self.buffer_size)
        if size == 0:
            raise ValueError("Cannot sample from empty buffer")
        samples = min(samples, size)

        # Anneal beta per sample() call (a sample precedes each gradient step).
        self._sample_calls += 1
        if self._sample_calls % self.beta_update_freq == 0:
            self.update_beta()
 
        if self.priority == "proportional":
            indices, probs, weights = self._sample_proportional(samples, size)
        else:
            indices, probs, weights = self._sample_rank(samples, size)

        # Clamp indices to current buffer size to ensure not sampling outside bounds
        indices = indices.clamp(max=size - 1)
 
        sample = {
            "states": tree_clone(tree_index(self.states, indices)),
            "actions": self.actions[indices].clone(),
            "rewards": self.rewards[indices].clone(),
            "intrinsic_rewards": self.intrinsic_rewards[indices].clone(),
            "next_states": tree_clone(tree_index(self.next_states, indices)),
            "terminations": self.terminations[indices].clone(),
            "truncations": self.truncations[indices].clone(),
            "trajectory_lengths": self.trajectory_lengths[indices].clone(),
            "raw_actions": self.raw_actions[indices].clone(),
            "log_probs": self.log_probs[indices].clone(),
            "weights": weights,
            "probs": probs,
            "indices": indices,
        }
        if self.initial_hidden is not None:
            sample["initial_hidden"] = tree_clone(tree_index(self.initial_hidden, indices))
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
        """Stratified sum-tree sampling with importance-sampling weights.

        Args:
            samples: Number of draws.
            size: Current filled buffer size (unused when the tree is empty;
                used only for the uniform fallback and IS weight formula).

        Returns:
            indices: Selected buffer indices.
            probs: Sampling probabilities ``priority / total``.
            weights: IS weights ``(size * probs)^(-beta)``, max-normalized.
        """
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
        """Stratified rank-based sampling with importance-sampling weights.

        Args:
            samples: Number of draws (also the number of rank segments).
            size: Current filled buffer size; ``_maybe_resort`` guarantees
                the sorted cache covers exactly this many entries.

        Returns:
            indices: Selected buffer indices.
            probs: Rank probabilities ``(rank+1)^(-alpha) / Z``.
            weights: IS weights ``(size * probs)^(-beta)``, max-normalized.
        """
        self._maybe_resort(size)
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
        """Resort ``priorities[:size]`` when the cache is missing or stale.

        Triggers on any of: no cache yet (``sorted_indices is None``), a
        coverage mismatch (``sorted_indices.numel() != size``, e.g. after
        growth or a ``reset``), or staleness (``_samples_since_sort >=
        sort_freq``). Post-condition: ``sorted_indices.numel() == size``.

        Args:
            size: Number of leading priority entries to sort.
        """
        needs_resort = (
            self.sorted_indices is None
            or self.sorted_indices.numel() != size
            or self._samples_since_sort >= self.sort_freq
        )

        if needs_resort:
            self.sorted_indices = T.argsort(self.priorities[:size], descending=True)
            self._samples_since_sort = 0

    def _get_segments(
        self, size: int, samples: int
    ) -> Tuple[T.Tensor, T.Tensor]:
        """Return cached ``(seg_starts, seg_ends)``, recomputing if needed.

        Args:
            size: Rank support size.
            samples: Number of equal-probability segments.

        Returns:
            seg_starts: Inclusive start ranks per segment.
            seg_ends: Exclusive end ranks per segment.
        """
        key = (size, samples)
        if self._seg_cache_key != key:
            self._compute_segments(size, samples)
            self._seg_cache_key = key
        return self._seg_starts, self._seg_ends

    def _compute_segments(self, size: int, k: int) -> None:
        """Compute k equal-probability segment boundaries over the rank PMF.

        Uses ``P(r) = (r+1)^(-α) / Z`` for ``r = 0..size-1`` and stores
        ``_seg_starts``, ``_seg_ends``, and ``_rank_Z``.

        Args:
            size: Rank support size.
            k: Number of segments (typically the sample count).
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
        """Linearly anneal β from β_start toward 1.0 over beta_iter updates."""
        self._beta_updates += 1
        progress = min(self._beta_updates / self.beta_iter, 1.0)
        self.beta = self.beta_start + progress * (1.0 - self.beta_start)
 
    def update_priorities(self, indices: T.Tensor, td_errors: T.Tensor) -> None:
        """Recompute priorities from TD errors for previously sampled indices.

        Stores ``p = (|δ| + ε)^α``. NaN priorities are replaced by the batch
        mean of valid entries (or ``ε^α`` if none).

        Args:
            indices: Buffer indices returned by the last ``sample``.
            td_errors: TD errors aligned with ``indices`` (flattened if needed
                by the caller).
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

    def reset(self) -> None:
        """Zero transition storage and all prioritized-sampling state.

        Calls ``ReplayBuffer.reset`` for the transition tensors, then clears
        whichever priority backend is active so a cleared buffer does not
        keep sampling stale leaves / ranks from before the reset: the
        ``SumTree`` in proportional mode, or ``priorities``,
        ``sorted_indices``, ``max_priority_rank``, and the segment cache in
        rank mode. Also zeros ``_samples_since_sort``.

        Deliberately leaves ``beta``, ``_sample_calls``, and
        ``_beta_updates`` untouched: the β-annealing schedule tracks
        gradient steps, not buffer contents, so it should not restart just
        because the buffer was cleared.
        """
        super().reset()
        self._samples_since_sort = 0
        if self.priority == "proportional":
            self.sum_tree.reset()
        else:  # rank
            self.priorities.zero_()
            self.sorted_indices = None
            self.max_priority_rank = T.tensor(1.0, dtype=T.float32, device=self.device)
            self._seg_cache_key = (-1, -1)
            self._seg_starts = None
            self._seg_ends = None
            self._rank_Z = T.tensor(0.0, dtype=T.float32, device=self.device)

    def get_config(self) -> Dict[str, Any]:
        """Extend the replay config with PER hyperparameters.

        Returns:
            Parent config plus ``alpha``, ``beta_start``, ``beta_iter``,
                ``beta_update_freq``, ``priority``, ``sort_freq``, and
                ``epsilon``.
        """
        config = super().get_config()
        config["type"] = self.__class__.__name__
        config["config"].update({
            "alpha": self.alpha,
            "beta_start": self.beta_start,
            "beta_iter": self.beta_iter,
            "beta_update_freq": self.beta_update_freq,
            "priority": self.priority,
            "sort_freq": self.sort_freq,
            "epsilon": self.epsilon,
        })
        return config

class RolloutBuffer(Buffer):
    """On-policy buffer storing fixed-horizon rollouts per environment."""

    def __init__(
        self,
        env: EnvWrapper,
        buffer_size: int,
        hindsight: HindsightRelabeler | None = None,
        device: str | T.device | None = None
    ):
        """Allocate ``(buffer_size, num_envs, ...)`` rollout tensors.

        Args:
            env: Wrapped environment (``num_envs`` sizes the second axis).
            buffer_size: Horizon length (timesteps) stored per env.
            hindsight: Optional HER relabeler (unused by this class's
                ``record`` / ``add`` path; subclasses may use it).
            device: Torch device; ``None`` resolves via ``get_device``.
        """
        super().__init__(env, buffer_size, hindsight, device)
        self.cur_idx = T.zeros((env.num_envs,), dtype=T.long, device=self.device)

        #Instantiate buffers (states/next_states: single tensor or per-key dict)
        self.states = alloc_from_spec(self.obs_spec, (buffer_size, env.num_envs), self.device)
        self.actions = T.zeros((buffer_size, env.num_envs, *self.action_space_shape), dtype=self.action_type, device=self.device)
        self.raw_actions = T.zeros((buffer_size, env.num_envs, *self.action_space_shape), dtype=self.action_type, device=self.device)
        self.log_probs = T.zeros((buffer_size, env.num_envs), dtype=T.float32, device=self.device)
        self.rewards = T.zeros((buffer_size, env.num_envs), dtype=T.float32, device=self.device)
        self.intrinsic_rewards = T.zeros((buffer_size, env.num_envs), dtype=T.float32, device=self.device)
        self.next_states = alloc_from_spec(self.obs_spec, (buffer_size, env.num_envs), self.device)
        self.terminations = T.zeros((buffer_size, env.num_envs), dtype=T.bool, device=self.device)
        self.truncations = T.zeros((buffer_size, env.num_envs), dtype=T.bool, device=self.device)
        # Tracks initial steps of each environment (Phantom steps)
        self.first_steps = T.zeros((buffer_size, env.num_envs), dtype=T.bool, device=self.device)
        
        if self.env.goal_key is not None and self.goal_space_shape is not None:
            self.desired_goals = T.zeros((buffer_size, env.num_envs, *self.goal_space_shape), dtype=T.float32, device=self.device)
            self.state_achieved_goals = T.zeros((buffer_size, env.num_envs, *self.goal_space_shape), dtype=T.float32, device=self.device)
            self.next_state_achieved_goals = T.zeros((buffer_size, env.num_envs, *self.goal_space_shape), dtype=T.float32, device=self.device)

    def record(self, cur_observation: Observation, prev_observation: Observation, actions: Action, prev_dones: T.Tensor) -> None:
        """Unpack one vectorized step into ``add``.

        Passes ``prev_dones`` as ``first_steps`` so post-episode bootstrap
        steps can be masked out at sample time.

        Args:
            cur_observation: Current-step observation and rewards.
            prev_observation: Previous-step observation (states / goals).
            actions: Actions taken from ``prev_observation``.
            prev_dones: Per-env done flags from the previous step.
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
        """Write one timestep for every env at ``cur_idx``, then increment.

        Args:
            states: Per-env states at this step.
            actions: Per-env actions.
            rewards: Per-env rewards.
            next_states: Per-env next states.
            terminations: Per-env termination flags.
            truncations: Per-env truncation flags.
            raw_actions: Optional raw actions.
            log_probs: Optional action log-probabilities.
            intrinsic_rewards: Optional intrinsic rewards.
            state_achieved_goals: Required when the env has a goal key.
            next_state_achieved_goals: Required when the env has a goal key.
            desired_goals: Required when the env has a goal key.
            first_steps: Per-env mask marking the first step after a done
                (phantom / bootstrap step); defaults to all-False.
        """
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

        
        # Add values to buffers at indices (tree_assign casts each leaf to its
        # storage dtype/device — dict observations write per key)
        self._sync_storage_dtypes(states)
        tree_assign(self.states, (idx, env_ids), states)
        self.actions[idx, env_ids] = actions.to(device=self.device, dtype=self.action_type)
        self.rewards[idx, env_ids] = rewards.to(device=self.device, dtype=T.float32)
        if intrinsic_rewards is not None:
            self.intrinsic_rewards[idx, env_ids] = intrinsic_rewards.to(device=self.device, dtype=T.float32)
        tree_assign(self.next_states, (idx, env_ids), next_states)
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
        """Return all stored timesteps up to ``cur_idx``, then reset indices.

        Slices ``[:max(cur_idx)]`` for every env (rollouts share length).
        Includes ``first_steps`` and ``valid_indices`` (flattened non-phantom
        positions). Clears ``cur_idx`` / ``first_steps`` via ``reset`` but
        does not zero the underlying data tensors.

        Args:
            **kwargs: Ignored; accepted for interface symmetry with
                off-policy ``sample(samples=...)``.

        Returns:
            Dict of cloned rollout tensors plus ``valid_indices``.
        """
        idx = int(self.cur_idx.max().item())
        if idx <= 0:
            raise ValueError("Cannot sample from empty buffer")

        # Create tensor of non first step (Phantom) indices for valid training samples
        first_steps = self.first_steps[:idx].clone()
        valid_indices = (first_steps.reshape(-1) == 0).nonzero()
        sample = {
            "states": tree_clone(tree_index(self.states, slice(None, idx))),
            "actions": self.actions[:idx].clone(),
            "rewards": self.rewards[:idx].clone(),
            "intrinsic_rewards": self.intrinsic_rewards[:idx].clone(),
            "next_states": tree_clone(tree_index(self.next_states, slice(None, idx))),
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
        """Zero per-env write indices and the ``first_steps`` mask."""
        self.cur_idx.zero_()
        self.first_steps.zero_()

    def is_ready(self, **kwargs: Any) -> bool:
        """Always ``True``; on-policy rollouts are sampled when the trainer asks.

        Args:
            **kwargs: Ignored; accepted for interface symmetry.
        """
        return True
    
    def get_config(self) -> Dict[str, Any]:
        """Return the base buffer config with this class's ``type``."""
        config = super().get_config()
        config['type'] = self.__class__.__name__
        return config

class TrajectoryBuffer(RolloutBuffer):
    """On-policy buffer that collects completed episode trajectories."""

    def __init__(
        self,
        env: EnvWrapper,
        buffer_size: int,
        hindsight: HindsightRelabeler | None = None,
        device: Optional[str] = None
    ):
        """Allocate rollout storage plus a list of completed trajectories.

        When ``hindsight`` is set, it must use ``output_format='flat'``.

        Args:
            env: Wrapped environment.
            buffer_size: Max timesteps retained per env while an episode is
                in progress (same layout as ``RolloutBuffer``).
            hindsight: Optional HER relabeler (``flat`` output only).
            device: Torch device string; ``None`` resolves via ``get_device``.
        """
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
        """Unpack one vectorized step into ``TrajectoryBuffer.add``.

        Args:
            cur_observation: Current-step observation and rewards.
            prev_observation: Previous-step observation (states / goals).
            actions: Actions taken from ``prev_observation``.
            prev_dones: Per-env done flags from the previous step.
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
        """Write a step, then harvest finished episodes into the trajectory list.

        After ``RolloutBuffer.add``, any env whose latest stored step is done
        contributes a trajectory of non-phantom steps to
        ``completed_trajectories`` (and optional HER flat copies), then that
        env's ``cur_idx`` is reset to 0.

        Args:
            states: Per-env states at this step.
            actions: Per-env actions.
            rewards: Per-env rewards.
            next_states: Per-env next states.
            terminations: Per-env termination flags.
            truncations: Per-env truncation flags.
            raw_actions: Optional raw actions.
            log_probs: Optional action log-probabilities.
            intrinsic_rewards: Optional intrinsic rewards.
            state_achieved_goals: Required when the env has a goal key.
            next_state_achieved_goals: Required when the env has a goal key.
            desired_goals: Required when the env has a goal key.
            first_steps: Per-env phantom-step mask (see ``RolloutBuffer.add``).
        """
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
                "states": tree_clone(tree_index(tree_index(self.states, (slice(None, idx), i)), valid_steps)),
                "actions": self.actions[:idx, i][valid_steps].clone(),
                "rewards": self.rewards[:idx, i][valid_steps].clone(),
                "intrinsic_rewards": self.intrinsic_rewards[:idx, i][valid_steps].clone(),
                "next_states": tree_clone(tree_index(tree_index(self.next_states, (slice(None, idx), i)), valid_steps)),
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
        """Return and clear the list of completed trajectories.

        Does not reset in-progress per-env write indices.

        Args:
            **kwargs: Ignored; accepted for interface symmetry.

        Returns:
            Shallow copy of ``completed_trajectories`` collected since the
                last sample (list is then emptied).
        """
        trajectories = self.completed_trajectories[:]
        # Clear trajectories
        self.completed_trajectories = []
        return trajectories

    def reset(self) -> None:
        """Clear write indices and discard any completed trajectories."""
        super().reset()
        self.completed_trajectories = []