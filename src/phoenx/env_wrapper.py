"""Gymnasium, EnvPool, and Isaac Lab environment adapters and helpers.

Adapters under ``EnvWrapper`` (``GymnasiumWrapper``, ``EnvPoolWrapper``,
``IsaacSimWrapper``) present a shared reset/step surface that returns
``Observation``. ``Action`` is the caller-supplied counterpart for steps.
Observation and action helpers
(one-hot encoding, NumPy↔Torch conversion) sit alongside ``WRAPPER_REGISTRY``,
which resolves the wrapper names a config may list. The ``VectorNStepReward``
n-step collector attaches sliding trajectory windows to
``info['n-step trajectory']`` for buffers that consume them. JSON helpers at
the bottom serialize ``EnvSpec`` / ``WrapperSpec`` and ``GymnasiumWrapper``
config for logging and checkpoints.
"""

import json
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING
from abc import abstractmethod
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
from .obs_utils import (
    flatten_obs, tree_assign, tree_cat, tree_index, tree_map, tree_stack,
)
if TYPE_CHECKING:
    from .intrinsic_motivation import IntrinsicMotivation


@dataclass
class Observation:
    """Packed environment transition returned by adapter ``reset`` / ``step``.

    Attributes:
        states: Observation tensor (or multi-modal dict of tensors).
        goals: Desired-goal tensor when goal-conditioned, else ``None``.
        ach_goals: Achieved-goal tensor when goal-conditioned, else ``None``.
        rewards: Step rewards; ``None`` on ``reset``.
        intrinsic_rewards: Intrinsic rewards when an IM module is attached.
        terminations: Episode termination flags; ``None`` on ``reset``.
        truncations: Episode truncation flags; ``None`` on ``reset``.
        n_step_trajectory: N-step window dict popped from infos when present.
        infos: Raw info dict from the underlying env.
    """

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
    """Packed action batch passed into adapter ``step`` / ``VectorNStepReward``.

    Attributes:
        actions: Env-facing action tensor (after any squashing / discretization).
        raw_actions: Pre-transform actions when the policy emits them separately.
        log_probs: Per-env log-probabilities of ``actions``, or ``None``.
        hidden: Recurrent state *before* this step's forward, flattened to
            batch-first tensors via ``ModularModel.hidden_to_tensors``
            (``None`` for feedforward agents). ``VectorNStepReward`` rings this
            so each emitted n-step window carries the hidden at its first step
            (R2D2 stored state).
    """

    actions: T.Tensor
    raw_actions: T.Tensor | None = None
    log_probs: T.Tensor | None = None
    hidden: dict | None = None

class VectorNStepReward(VectorWrapper):
    """Vectorized ring-buffer n-step trajectory collector.

    Maintains a per-env ring of length ``n`` for states, actions, rewards,
    terminations/truncations, optional goals, intrinsic rewards, and R2D2
    stored recurrent state. Each ``step`` writes into ``info['n-step
    trajectory']`` via ``_build_trajectories``: envs with ``length > 0`` emit
    windows of shape ``(num_valid, n, *)``, with repeat-padding for
    states/actions and zero-padding for rewards/flags. On terminal steps,
    trailing sub-windows are flushed so every in-episode step becomes an
    anchor. Autoreset: envs whose previous step was done skip the write
    (``prev_done``), matching Gymnasium vector autoreset semantics. Call
    ``set_action`` before ``step`` so raw actions, log-probs, and hidden
    state are recorded; optionally ``set_intrinsic_motivation`` for rollout
    intrinsic rewards.
    """

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
        """Allocate per-env ring pointers; buffers are sized on the first step.

        Args:
            env: Vector environment to wrap.
            n: Ring length (max trajectory window size).
            obs_key: Dict key for the agent observation, or ``None`` to use
                the full observation (minus goal keys when set).
            goal_key: Dict key for desired goals, or ``None`` if unused.
            ach_goal_key: Dict key for achieved goals, or ``None`` if unused.
            log_level: Logger level name (uppercased).
            name: Logger name; defaults to the class name.
            **kwargs (Any): Extra attributes set on ``self`` (e.g. diagnostic
                knobs such as ``_diag_freq``).
        """
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
        # Per-step recurrent state ring (R2D2 stored state), flattened tensors
        self._buf_hidden = None

        self.current_states = None
        self.current_action = None

        # Helper index tensors
        self._t_idx = T.arange(self.n, device=self.device)
        self._env_idx = T.arange(self.num_envs, device=self.device)
        self._env_idx_nx1 = self._env_idx.unsqueeze(1).expand(self.num_envs, self.n)

    def set_action(self, action: Action) -> None:
        """Store the ``Action`` for the upcoming ``step`` write.

        Args:
            action: Actions, optional raw actions / log-probs / hidden state.
        """
        self.current_action = action

    def set_intrinsic_motivation(self, intrinsic_motivation: "IntrinsicMotivation") -> None:
        """Attach an intrinsic-motivation module for per-step rollout rewards.

        Args:
            intrinsic_motivation: Module whose ``compute_rollout_reward`` is
                called inside ``step``, or any object with that API.
        """
        self.intrinsic_motivation = intrinsic_motivation
    
    def reset(self, **kwargs):
        """Reset the vector env and clear ring pointers.

        Args:
            **kwargs (Any): Forwarded to ``env.reset``.

        Returns:
            states (Any): Observation batch from the underlying ``reset``.
            infos (dict): Info dict with an empty ``n-step trajectory`` entry.
        """
        states, infos = self.env.reset(**kwargs)
        self.head.zero_()
        self.length.zero_()
        self.prev_done.zero_()
        self.current_states = states
        infos.setdefault('n-step trajectory', {})
        return states, infos

    def _alloc_like(self, sample) -> T.Tensor | dict:
        """Allocate a ``(num_envs, n, *tail)`` zero buffer matching ``sample``.

        Dict observations allocate per key, preserving each modality's dtype.

        Args:
            sample (Any): Tensor or dict-of-tensors with leading ``num_envs``.

        Returns:
            Zero-initialized buffer tree on ``self.device``.
        """
        def _alloc(t: T.Tensor) -> T.Tensor:
            tail = tuple(t.shape[1:])
            return T.zeros((self.num_envs, self.n, *tail), dtype=t.dtype, device=self.device)
        return tree_map(_alloc, sample)

    def _extract_state(self, states):
        """Select the per-step observation payload (tensor or non-goal dict).

        Args:
            states (Any): Raw observation batch (array, tensor, or dict).

        Returns:
            Tensor or dict-of-tensors on ``self.device``, excluding goal keys
            when ``obs_key`` is unset.
        """
        if self.obs_key is not None:
            payload = states[self.obs_key]
        elif isinstance(states, dict):
            excluded = {k for k in (self.goal_key, self.ach_goal_key) if k}
            payload = {k: v for k, v in states.items() if k not in excluded}
        else:
            payload = states
        return tree_map(lambda v: T.as_tensor(v, device=self.device), payload)

    def step(self, actions: T.Tensor):
        """Step the vector env, append to rings, and emit trajectory infos.

        Lazily allocates buffers on the first call. Skips advancing the ring
        for envs that terminated on the previous step (autoreset). Writes
        ``infos['n-step trajectory']`` from ``_build_trajectories``, then
        clears head/length for newly done envs.

        Args:
            actions: Action batch for the underlying vector env.

        Returns:
            next_states (Any): Next observation batch.
            rewards (torch.Tensor): Env rewards as a device tensor.
            terminations (torch.Tensor): Termination flags.
            truncations (torch.Tensor): Truncation flags.
            infos (dict): Infos including ``n-step trajectory``.
        """
        self._step += 1
        next_states, rewards, terminations, truncations, infos = self.env.step(actions)

        # Ensure tensors (log_probs normalized to float32 so the ring dtype is
        # agent-independent)
        actions = T.as_tensor(actions, device=self.device)
        raw_actions = T.as_tensor(self.current_action.raw_actions, device=self.device) if self.current_action.raw_actions is not None else T.zeros_like(actions)
        rewards = T.as_tensor(rewards, device=self.device)
        log_probs = (T.as_tensor(self.current_action.log_probs, device=self.device).float()
                     if self.current_action.log_probs is not None
                     else T.zeros(self.num_envs, dtype=T.float32, device=self.device))
        terminations = T.as_tensor(terminations, device=self.device)
        truncations = T.as_tensor(truncations, device=self.device)
        dones = T.logical_or(terminations, truncations)

        state_b = self._extract_state(self.current_states)
        next_state_b = self._extract_state(next_states)
        if self.goal_key:
            goals_b = T.as_tensor(self.current_states[self.goal_key], device=self.device)
            ach_goals_b = T.as_tensor(self.current_states[self.ach_goal_key], device=self.device)
            next_ach_goals_b = T.as_tensor(next_states[self.ach_goal_key], device=self.device)

        if self.intrinsic_motivation is not None:
            with T.no_grad():
                intrinsic_rewards = self.intrinsic_motivation.compute_rollout_reward(
                    flatten_obs(state_b), flatten_obs(next_state_b), actions, env_indices = T.arange(self.env.num_envs, device=self.intrinsic_motivation.device)
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

        # Recurrent stored-state ring: (re)allocate whenever the hidden schema
        # appears or changes (e.g. a different agent reuses the env).
        # getattr: tolerate duck-typed Action stand-ins without the field.
        action_hidden = getattr(self.current_action, 'hidden', None)
        if action_hidden is not None:
            incoming = {k: T.as_tensor(v, device=self.device)
                        for k, v in action_hidden.items()}
            if self._buf_hidden is None or set(self._buf_hidden) != set(incoming):
                self._buf_hidden = self._alloc_like(incoming)

        # Only envs whose previous step was NOT terminal get a new entry appended.
        active = ~self.prev_done
        write_pos = self.head

        env_idx = self._env_idx
        tree_assign(self._buf_states, (env_idx, write_pos), state_b)
        tree_assign(self._buf_next_states, (env_idx, write_pos), next_state_b)
        self._buf_actions[env_idx, write_pos] = actions
        self._buf_raw_actions[env_idx, write_pos] = raw_actions
        self._buf_log_probs[env_idx, write_pos] = log_probs
        self._buf_rewards[env_idx, write_pos] = rewards
        self._buf_intrinsic_rewards[env_idx, write_pos] = intrinsic_rewards
        self._buf_terminations[env_idx, write_pos] = terminations
        self._buf_truncations[env_idx, write_pos] = truncations
        if self._buf_hidden is not None and action_hidden is not None:
            tree_assign(self._buf_hidden, (env_idx, write_pos),
                        {k: T.as_tensor(v, device=self.device)
                         for k, v in action_hidden.items()})
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
        """Gather per-env ring windows into a batched trajectory dict.

        Emits tensors of shape ``(num_valid_envs, n, *)`` for envs with
        ``length > 0``. Positions beyond each env's ``length`` use
        repeat-padding for states / next_states / actions (and goal /
        log-prob fields) and zero-padding for rewards / terminations /
        truncations / intrinsic rewards. When ``dones`` marks terminals,
        also appends flushed tail windows so every in-episode step is an
        anchor. Returns ``None`` when no env has data.

        Args:
            dones: Per-env done flags for the current step; when any are
                true, terminal tails are flushed into the returned dict.

        Returns:
            Trajectory dict with leading dim ``num_valid`` (plus flushed
            tails), or ``None`` if every env has ``length == 0``.
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
        states_all = tree_index(self._buf_states, (env_idx, gather_idx))
        next_states_all = tree_index(self._buf_next_states, (env_idx, gather_idx))
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
            'states': tree_index(states_all, valid),
            'actions': actions_all[valid],
            'rewards': rewards_all[valid],
            'next_states': tree_index(next_states_all, valid),
            'terminations': terminations_all[valid],
            'truncations': truncations_all[valid],
            'raw_actions': raw_actions_all[valid],
            'log_probs': log_probs_all[valid],
            'intrinsic_rewards': intrinsic_rewards_all[valid],
            'trajectory_lengths': length[valid],
        }
        if self._buf_hidden is not None:
            # R2D2 stored state: the recurrent state at each window's FIRST step.
            trajectory['initial_hidden'] = {
                k: buf[env_idx[:, 0], start][valid]
                for k, buf in self._buf_hidden.items()
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
            tail_rows: dict[str, list] = {
                k: [] for k in (*rep_fields, *zero_fields)
            }
            tail_hidden_rows: list[dict] = []
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
                        # states/next_states may be dict-of-tensors (multi-modal)
                        tail_rows[key].append(tree_index(arr, (e, idx)))
                    for key, arr in zero_fields.items():
                        row = arr[e, idx].clone()
                        # Re-zero padding: required when L == n, where the
                        # clamped indices would otherwise repeat a real entry
                        # (e.g. the terminal reward/flag) into padding slots.
                        row[~new_valid] = 0
                        tail_rows[key].append(row)
                    if self._buf_hidden is not None:
                        # Tail window starts j steps into the episode window.
                        ring_pos = (start[e] + j) % n
                        tail_hidden_rows.append(
                            {k: buf[e, ring_pos] for k, buf in self._buf_hidden.items()})
                    tail_lengths.append(new_len)
            if tail_lengths:
                for key in tail_rows:
                    stacked = tree_stack(tail_rows[key], dim=0)  # [num_tail, n, ...]
                    trajectory[key] = tree_cat([trajectory[key], stacked], dim=0)
                if self._buf_hidden is not None and tail_hidden_rows:
                    stacked_hidden = tree_stack(tail_hidden_rows, dim=0)
                    trajectory['initial_hidden'] = tree_cat(
                        [trajectory['initial_hidden'], stacked_hidden], dim=0)
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
    """Map a Discrete observation to a float32 one-hot ``Box`` vector."""

    def __init__(self, env):
        """Replace a Discrete observation space with a one-hot ``Box``.

        Args:
            env (gym.Env): Environment whose observation space must be Discrete.
        """
        super().__init__(env)
        assert isinstance(self.observation_space, gym.spaces.Discrete), "Observation space must be Discrete."
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.observation_space.n,), dtype=np.float32)
    
    def observation(self, obs):
        """Encode a discrete index as a one-hot float32 vector.

        Args:
            obs (Any): Integer Discrete observation.

        Returns:
            one_hot (np.ndarray): Float32 vector of length ``n`` with a single 1.
        """
        one_hot = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        one_hot[obs] = 1.0
        return one_hot

class NumpyToTorch(VectorWrapper):
    """Convert vector-env observations and actions between NumPy and Torch."""

    def __init__(self, env, device=None):
        """Wrap a vector env and store the Torch device for conversions.

        Args:
            env (VectorEnv): Vector environment to wrap.
            device (torch.device | str | None): Device for ``to_torch``; ``None``
                leaves device selection to ``to_torch``.
        """
        super().__init__(env)
        self.device = device

    def reset(self, *, seed=None, options=None):
        """Reset with NumPy options; return Torch observation and info.

        Args:
            seed (int | None): Seed forwarded to ``env.reset``.
            options (Any): Options converted via ``to_numpy`` before reset.

        Returns:
            observation (Any): Observation batch as Torch tensors.
            info (Any): Info tree as Torch tensors.
        """
        obs, info = self.env.reset(seed=seed, options=to_numpy(options))
        return to_torch(obs, self.device), to_torch(info, self.device)

    def step(self, actions):
        """Step with NumPy actions; return Torch transition fields.

        Args:
            actions (Any): Action batch converted via ``to_numpy`` before step.

        Returns:
            observation (Any): Next observation as Torch tensors.
            reward (Any): Rewards as Torch tensors.
            terminated (Any): Termination flags as Torch tensors.
            truncated (Any): Truncation flags as Torch tensors.
            info (Any): Info tree as Torch tensors.
        """
        obs, reward, terminated, truncated, info = self.env.step(to_numpy(actions))
        return (
            to_torch(obs, self.device),
            to_torch(reward, self.device),
            to_torch(terminated, self.device),
            to_torch(truncated, self.device),
            to_torch(info, self.device),
        )

    def render(self):
        """Forward ``render`` to the wrapped vector env.

        Returns:
            frame (Any): Render output from the underlying env.
        """
        return self.env.render()

class VectorOneHotObservation(VectorWrapper):
    """Vectorized one-hot encoding for Discrete observation spaces."""

    def __init__(self, env):
        """Require a Discrete ``single_observation_space`` and cache ``n``.

        Args:
            env (VectorEnv): Vector env whose single observation space is Discrete.
        """
        super().__init__(env)
        assert isinstance(self.single_observation_space, gym.spaces.Discrete)
        self._n = self.single_observation_space.n

    def reset(self, **kwargs):
        """Reset and one-hot encode the observation batch.

        Args:
            **kwargs (Any): Forwarded to ``env.reset``.

        Returns:
            observation (np.ndarray): One-hot float32 array of shape
                ``(batch, n)``.
            info (Any): Info from the underlying ``reset``.
        """
        obs, info = self.env.reset(**kwargs)
        return self._encode(obs), info

    def step(self, actions):
        """Step and one-hot encode the next observation batch.

        Args:
            actions (Any): Action batch for the underlying vector env.

        Returns:
            observation (np.ndarray): One-hot float32 array of shape
                ``(batch, n)``.
            reward (Any): Rewards from the underlying ``step``.
            terminated (Any): Termination flags.
            truncated (Any): Truncation flags.
            info (Any): Info from the underlying ``step``.
        """
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

class EnvWrapper:
    """Abstract base class for environment wrappers.

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
        """Store environment id, vectorization size, goal keys, and seed.

        Args:
            cfg: Environment id or config string passed to the concrete adapter.
            num_envs: Number of parallel environments to create.
            obs_key: Dict-observation key for the agent state. When unset,
                remaining non-goal keys become multi-modal observations;
                required only for a list of per-step dicts.
            goal_key: Dict key for the desired goal, or ``None`` if unused.
            ach_goal_key: Dict key for the achieved goal, or ``None`` if unused.
            wrappers: Optional list of wrapper specs, each a dict with ``type``
                and optional ``params``.
            render_mode: Gymnasium render mode forwarded at construction, or
                ``None`` for no rendering.
            seed: RNG seed; when ``None``, a random 31-bit seed is drawn.
        """
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
        """Split observations, goals, and achieved goals into tensors on device.

        Args:
            states: Raw env observation: array, tensor, dict, or list of dicts.

        Returns:
            Observations as a tensor or, for multi-modal dicts, a dict of
                tensors on the active device.
            Goal tensor, or ``None`` when ``goal_key`` is unset.
            Achieved-goal tensor, or ``None`` when ``ach_goal_key`` is unset.
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
            if self.obs_key:
                obs = states.get(self.obs_key)
            else:
                # Multi-modal: keep every non-goal key as its own modality.
                excluded = {k for k in (self.goal_key, self.ach_goal_key) if k}
                obs = {k: v for k, v in states.items() if k not in excluded}
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

        if isinstance(obs, dict):
            # Preserve per-modality dtypes (uint8 images stay uint8; the model
            # casts/scales at its input boundary).
            obs = {
                k: (v if isinstance(v, T.Tensor) else T.as_tensor(v, device=device))
                for k, v in obs.items()
            }
        elif not isinstance(obs, T.Tensor):
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
        """Build a JSON-serializable config dict for this wrapper.

        Returns:
            config (dict): Mapping with ``type`` (class name) and a nested
                ``config`` of constructor kwargs.
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
        """Reset the environment to an initial state.

        Returns:
            observation (Observation): Initial observation of the environment.
        """
        pass
    
    @abstractmethod
    def step(self, action) -> Observation:
        """Take an action in the environment.

        Args:
            action (Any): Action batch for the (vectorized) environment.

        Returns:
            Observation dataclass with states, optional goals, rewards,
                terminations, truncations, and infos.
        """
        pass

    @abstractmethod
    def _initialize_env(self):
        """Initialize the underlying environment instance.

        Returns:
            env (Any): The initialized backend environment.
        """
        pass

    def clone(self, num_envs:int=1, **kwargs) -> 'EnvWrapper':
        """Create a new wrapper instance from this one's JSON config.

        Args:
            num_envs: Number of parallel environments for the clone.
            **kwargs (Any): Constructor kwargs that override values from the
                serialized config before reconstruction.

        Returns:
            New ``EnvWrapper`` instance built via ``from_json``.
        """
        config = json.loads(self.to_json())
        config['config'].update(num_envs=num_envs, **kwargs)
        return self.from_json(json.dumps(config))

    @abstractmethod
    def format_actions(self, actions: np.ndarray | T.Tensor):
        """Format actions for the environment.

        Args:
            actions: Actions to format.

        Returns:
            formatted (Any): Actions reshaped or converted for ``step``.
        """
        pass
    
    @property
    @abstractmethod
    def observation_space(self):
        """Get the observation space of the environment.

        Returns:
            space (gymnasium.spaces.Space): The (vector) observation space.
        """
        pass
    
    @property
    @abstractmethod
    def action_space(self):
        """Get the action space of the environment.

        Returns:
            space (gymnasium.spaces.Space): The (vector) action space.
        """
        pass

    @property
    def single_action_space(self):
        """Get the single-env action space for vectorized environments.

        Returns:
            space (gymnasium.spaces.Space): Action space of one sub-environment.
        """
        pass

    @property
    def single_observation_space(self):
        """Get the single-env observation space for vectorized environments.

        Returns:
            space (gymnasium.spaces.Space): Observation space of one
                sub-environment.
        """
        pass

    @abstractmethod
    def to_json(self) -> str:
        """Serialize the environment wrapper configuration to JSON.

        Returns:
            JSON string representing the environment configuration.
        """
        pass

    @classmethod
    def from_json(cls, json_string: str):
        """Create an environment wrapper instance from a JSON string.

        Delegates to the subclass ``from_json`` matching the ``type`` field in
        the JSON (``gymnasium``, ``envpool``, or ``isaacsim``).

        Args:
            json_string: JSON string representing the environment configuration.

        Returns:
            wrapper (EnvWrapper): A new environment wrapper instance.

        Raises:
            ValueError: If the type in the JSON is not recognized or if
                instantiation fails.
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
    """Wrapper for Gymnasium environments with additional utilities.

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
        """Build a Gymnasium vector env and apply configured wrappers.

        Args:
            cfg: Gymnasium environment id passed to ``gym.make_vec``.
            num_envs: Number of parallel environments to create.
            obs_key: Dict-observation key for the agent state. When unset,
                remaining non-goal keys become multi-modal observations;
                required only for a list of per-step dicts.
            goal_key: Dict key for the desired goal, or ``None`` if unused.
            ach_goal_key: Dict key for the achieved goal, or ``None`` if unused.
            wrappers: Optional list of wrapper specs, each a dict with ``type``
                and optional ``params``, applied as single-env or vector-aware
                wrappers.
            render_mode: Gymnasium render mode forwarded to ``make_vec``, or
                ``None`` for no rendering.
            seed: RNG seed; when ``None``, a random 31-bit seed is drawn by
                the base class.
        """
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
        """Create the Gymnasium vector env, apply wrappers, and torch-cast outputs.

        Resolves each entry in ``self.wrappers`` from ``WRAPPER_REGISTRY`` or
        built-in Gymnasium wrapper modules, builds a sync vector env via
        ``gym.make_vec``, then wraps it with ``NumpyToTorch``.

        Returns:
            env (gymnasium.vector.VectorEnv): Initialized vectorized environment
                with torch observation conversion.
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
        """Render one frame from the first sub-environment.

        Returns:
            RGB or other render array from ``env.render()`` index ``0``.
        """
        frame = self.env.render()        
        return frame[0]
        

    def reset(self, seed:int|None=None):
        """Reset the vector env and return an ``Observation``.

        Args:
            seed: Seed for ``env.reset`` and the action space; defaults to
                ``self.seed`` when omitted.

        Returns:
            observation (Observation): Initial states, optional goals, and
                infos; may attach ``n_step_trajectory`` when present in infos.
        """
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
        """Step the vector env and pack the transition into an ``Observation``.

        Args:
            action (Any): Action batch accepted by the underlying vector env.

        Returns:
            Observation with states, rewards, terminations, truncations, and
                infos; may attach ``n_step_trajectory`` when present in infos.
        """
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
        """Sample a random action and step once, returning the ``Observation``.

        Returns:
            observation (Observation): Result of ``step`` on a sampled action.
        """
        actions = self.action_space.sample()
        return self.step(actions)
    
    def format_actions(self, actions: np.ndarray | T.Tensor):
        """Convert actions to NumPy and reshape for the action space.

        Box actions become ``(num_envs, action_dim)``; Discrete or MultiDiscrete
        actions are raveled to 1-D.

        Args:
            actions: Action array or tensor to format.

        Returns:
            formatted (numpy.ndarray): Actions shaped for the vector env, or
                ``None`` when the action space type is not handled.
        """
        if isinstance(actions, T.Tensor):
            actions = actions.cpu().numpy()
        if isinstance(self.action_space, gym.spaces.Box):
            num_envs = self.env.num_envs
            num_actions = self.action_space.shape[-1]
            return actions.reshape(num_envs, num_actions)
        if isinstance(self.action_space, gym.spaces.Discrete) or isinstance(self.action_space, gym.spaces.MultiDiscrete):
            return actions.ravel()
    
    def close(self):
        """Close the underlying Gymnasium environment."""
        self.env.close()
    
    @property
    def observation_space(self):
        """Get the observation space of the environment.

        Returns:
            space (gymnasium.spaces.Space): The vector observation space.
        """
        return self.env.observation_space
    
    @property
    def action_space(self):
        """Get the action space of the environment.

        Returns:
            space (gymnasium.spaces.Space): The vector action space.
        """
        return self.env.action_space
    
    @property
    def single_action_space(self):
        """Get the single-env action space for vectorized environments.

        Returns:
            space (gymnasium.spaces.Space): Action space of one sub-environment.
        """
        return self.env.single_action_space

    @property
    def single_observation_space(self):
        """Get the single-env observation space for vectorized environments.

        Returns:
            space (gymnasium.spaces.Space): Observation space of one
                sub-environment.
        """
        return self.env.single_observation_space

    @property
    def finite_horizon(self)->bool:
        """Return whether the environment has a finite episode horizon.

        True when the base env spec sets ``max_episode_steps``, or when a
        ``TimeLimit`` wrapper is present in the wrap stack.
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
        """Build a JSON-serializable config with type ``gymnasium``.

        Returns:
            config (dict): Parent config with ``type`` set to ``"gymnasium"``.
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
        """Serialize the wrapper configuration to JSON.

        Returns:
            json_string (str): JSON encoding of ``self.config``.
        """
        return json.dumps(self.config)

    @classmethod
    def from_json(cls, json_env_spec):
        """Create a Gymnasium wrapper instance from a JSON string.

        Args:
            json_env_spec (str): JSON string with a nested ``config`` object of
                constructor kwargs.

        Returns:
            wrapper (GymnasiumWrapper): A new Gymnasium wrapper instance.

        Raises:
            ValueError: If construction from the parsed config fails.
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
        """Store the EnvPool env and batch its spaces for ``num_envs``.

        Args:
            envpool_env (Any): EnvPool gymnasium environment from
                ``envpool.make_gymnasium``.
            num_envs: Number of parallel sub-environments.
        """
        self._env = envpool_env
        self.num_envs = num_envs
        self.single_observation_space = envpool_env.observation_space
        self.single_action_space = envpool_env.action_space
        self.observation_space = utils.batch_space(envpool_env.observation_space, num_envs)
        self.action_space = utils.batch_space(envpool_env.action_space, num_envs)

    def reset(self, *, seed=None, options=None):
        """Reset the EnvPool env and return observation and info.

        Args:
            seed (Any): Accepted for VectorEnv API compatibility; not forwarded
                to EnvPool ``reset``.
            options (Any): Accepted for VectorEnv API compatibility; unused.

        Returns:
            obs_info (tuple): ``(observation, info)`` from the underlying env.
        """
        obs, info = self._env.reset()
        return obs, info

    def step(self, actions):
        """Step the EnvPool env with the given action batch.

        Args:
            actions (Any): Action batch accepted by the EnvPool env.

        Returns:
            step_result (Any): ``(obs, rewards, terminations, truncations,
                infos)`` from the underlying env.
        """
        return self._env.step(actions)

    def render(self, **kwargs):
        """Forward render kwargs to the EnvPool env.

        Args:
            **kwargs (Any): Keyword arguments passed to ``env.render``.

        Returns:
            frame (Any): Render output from the underlying env.
        """
        return self._env.render(**kwargs)

    def close(self):
        """Close the underlying EnvPool environment."""
        self._env.close()

    @property
    def spec(self):
        """Return the EnvPool env ``spec`` attribute if present.

        Returns:
            spec (Any): Env spec object, or ``None`` when absent.
        """
        return getattr(self._env, 'spec', None)

class EnvPoolWrapper(EnvWrapper):
    """Wrapper for EnvPool vectorized environments with PhoenX utilities.

    Builds an EnvPool gymnasium env via ``envpool.make_gymnasium``, maps a
    subset of Gymnasium wrappers to EnvPool constructor kwargs, adapts the
    result through ``EnvPoolAdapter``, then applies vector-aware wrappers and
    ``NumpyToTorch``.
    """

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
        """Build an EnvPool vector env and apply configured wrappers.

        Args:
            cfg: EnvPool task id passed as ``task_id`` to
                ``envpool.make_gymnasium``.
            num_envs: Number of parallel environments to create.
            obs_key: Dict-observation key for the agent state. When unset,
                remaining non-goal keys become multi-modal observations;
                required only for a list of per-step dicts.
            goal_key: Dict key for the desired goal, or ``None`` if unused.
            ach_goal_key: Dict key for the achieved goal, or ``None`` if unused.
            num_threads: EnvPool worker thread count; defaults to ``num_envs``
                when ``None``.
            wrappers: Optional list of wrapper specs, each a dict with ``type``
                and optional ``params``. Mapped types become EnvPool kwargs;
                vector-aware registry or gymnasium vector wrappers wrap the
                adapter; other types raise ``ValueError``.
            render_mode: Render mode forwarded to EnvPool when set, or ``None``.
            seed: RNG seed; when ``None``, a random 31-bit seed is drawn by
                the base class.
        """
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
        """Create the EnvPool env, adapt it, apply wrappers, and torch-cast outputs.

        Resolves each entry in ``self.wrappers`` via ``WRAPPER_TO_ENVPOOL_PARAM``,
        ``WRAPPER_REGISTRY`` (vector-aware only), or ``gymnasium.wrappers.vector``,
        builds the env with ``envpool.make_gymnasium``, wraps it in
        ``EnvPoolAdapter``, then applies remaining vector wrappers and
        ``NumpyToTorch``.

        Returns:
            env (Any): Initialized vectorized environment with torch observation
                conversion.

        Raises:
            ValueError: If a wrapper type is neither EnvPool-mapped nor
                vector-aware.
        """
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

    def render_frame(self)->np.ndarray:
        """Render one frame from the first sub-environment.

        Returns:
            RGB or other render array from ``env.render()`` index ``0``.
        """
        frame = self.env.render()        
        return frame[0]
        
    def reset(self, seed:int|None=None):
        """Reset the vector env and return an ``Observation``.

        Args:
            seed: Defaults to ``self.seed`` when omitted; reseeds
                ``self.env.action_space`` only (EnvPool is seeded at
                construction and the adapter does not forward this seed).

        Returns:
            observation (Observation): Initial states, optional goals, and
                infos; may attach ``n_step_trajectory`` when present in infos.
        """
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
        """Step the vector env and pack the transition into an ``Observation``.

        Args:
            action (Any): Action batch accepted by the underlying vector env.

        Returns:
            Observation with states, rewards, terminations, truncations, and
                infos; may attach ``n_step_trajectory`` when present in infos.
        """
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
        """Sample a random action, step once, and return states and goals only.

        Unlike a full ``step`` result, the returned ``Observation`` omits
        rewards, terminations, truncations, and infos.

        Returns:
            observation (Observation): States and optional goals from one
                sampled step.
        """
        actions = self.action_space.sample()
        observation = self.step(actions)
        obs, goals, ach_goals = self.extract_states_goals(observation.states)
        return Observation(
            states=obs,
            goals=goals,
            ach_goals=ach_goals
        )
    
    def format_actions(self, actions: np.ndarray | T.Tensor):
        """Convert actions to NumPy and reshape for the action space.

        Box actions become ``(num_envs, action_dim)``; Discrete or MultiDiscrete
        actions are raveled to 1-D.

        Args:
            actions: Action array or tensor to format.

        Returns:
            formatted (numpy.ndarray): Actions shaped for the vector env, or
                ``None`` when the action space type is not handled.
        """
        if isinstance(actions, T.Tensor):
            actions = actions.cpu().numpy()
        if isinstance(self.action_space, gym.spaces.Box):
            num_envs = self.env.num_envs
            num_actions = self.action_space.shape[-1]
            return actions.reshape(num_envs, num_actions)
        if isinstance(self.action_space, gym.spaces.Discrete) or isinstance(self.action_space, gym.spaces.MultiDiscrete):
            return actions.ravel()
    
    def close(self):
        """Close the underlying EnvPool environment."""
        self.env.close()
    
    @property
    def observation_space(self):
        """Get the observation space of the environment.

        Returns:
            space (gymnasium.spaces.Space): The vector observation space.
        """
        return self.env.observation_space
    
    @property
    def action_space(self):
        """Get the action space of the environment.

        Returns:
            space (gymnasium.spaces.Space): The vector action space.
        """
        return self.env.action_space
    
    @property
    def single_action_space(self):
        """Get the single-env action space for vectorized environments.

        Returns:
            space (gymnasium.spaces.Space): Action space of one sub-environment.
        """
        return self.env.single_action_space

    @property
    def single_observation_space(self):
        """Get the single-env observation space for vectorized environments.

        Returns:
            space (gymnasium.spaces.Space): Observation space of one
                sub-environment.
        """
        return self.env.single_observation_space

    @property
    def finite_horizon(self) -> bool:
        """Return whether the environment has a finite episode horizon.

        True when the env ``spec`` sets ``max_episode_steps`` to a non-``None``
        value; otherwise False.
        """
        spec = getattr(self.env, 'spec', None)
        if spec and hasattr(spec, 'max_episode_steps'):
            return spec.max_episode_steps is not None
        return False
    
    @property
    def config(self):
        """Build a JSON-serializable config with type ``envpool``.

        Returns:
            config (dict): Parent config with ``type`` set to ``"envpool"`` and
                ``num_threads`` added under the nested ``config``.
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
        """Serialize the wrapper configuration to JSON.

        Returns:
            json_string (str): JSON encoding of ``self.config``.
        """
        return json.dumps(self.config)

    @classmethod
    def from_json(cls, json_env_spec):
        """Create an EnvPool wrapper instance from a JSON string.

        Args:
            json_env_spec (str): JSON string with a nested ``config`` object of
                constructor kwargs.

        Returns:
            wrapper (EnvPoolWrapper): A new EnvPool wrapper instance.

        Raises:
            ValueError: If construction from the parsed config fails.
        """
        config = json.loads(json_env_spec)
        config = config['config']
        try:
            return cls(**config)
        except Exception as e:
            raise ValueError(f"Environment wrapper error: {config}, {e}")

_NEXT_STEP_ENV_CLS = None # cached after first build

def _get_next_step_env_cls():
    """Lazily build the NextStep ManagerBasedRLEnv subclass.

    Deferred until after the Omniverse app is launched, because importing
    ``isaaclab.envs`` (ManagerBasedRLEnv -> mdp -> controllers) requires a
    running Kit app. Keeping this out of module scope means non-Isaac runs
    never import or boot Isaac Sim.

    Returns:
        cls (type): Cached ``NextStepManagerBasedRLEnv`` subclass of
            Isaac Lab ``ManagerBasedRLEnv``.
    """
    global _NEXT_STEP_ENV_CLS
    if _NEXT_STEP_ENV_CLS is not None:
        return _NEXT_STEP_ENV_CLS
    try:
        from isaaclab.envs import ManagerBasedRLEnv  # type: ignore[reportMissingImports]
    except (ModuleNotFoundError, ImportError):
        from omni.isaac.lab.envs import ManagerBasedRLEnv  # type: ignore[reportMissingImports]
    class NextStepManagerBasedRLEnv(ManagerBasedRLEnv):
        """Converts a ManagerBasedRLEnv to use NextStep auto-reset mode."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._capture_terminal = False
            self._terminal_obs = None
            self._phantom_mask = T.zeros(self.num_envs, dtype=T.bool, device=self.device)
        def _reset_idx(self, env_ids):
            if self._capture_terminal and len(env_ids) > 0:
                self._terminal_obs = {k: v.clone() for k, v in self.observation_manager.compute(update_history=False).items()}
            super()._reset_idx(env_ids)
        def reset(self, *args, **kwargs):
            self._capture_terminal = False
            out = super().reset(*args, **kwargs)
            self._phantom_mask.zero_()
            self._terminal_obs = None
            self._capture_terminal = True
            return out
        def step(self, action):
            prev_term = self._phantom_mask
            self._terminal_obs = None
            states, rewards, terminations, truncations, extras = super().step(action)
            new_phantom = (terminations | truncations) & ~prev_term
            if self._terminal_obs is not None and new_phantom.any():
                ids = new_phantom.nonzero(as_tuple=False).squeeze(-1)
                for k, v in self._terminal_obs.items():
                    states[k][ids] = v[ids]
            if prev_term.any():
                ids = prev_term.nonzero(as_tuple=False).squeeze(-1)
                self._capture_terminal = False
                self._reset_idx(ids)
                self._capture_terminal = True
                reset_obs = self.observation_manager.compute(update_history=False)
                for k, v in reset_obs.items():
                    states[k][ids] = v[ids]
                # rewards[ids] = 0.0
                terminations[ids] = False
                truncations[ids] = False
            self._phantom_mask = new_phantom
            return states, rewards, terminations, truncations, extras
    _NEXT_STEP_ENV_CLS = NextStepManagerBasedRLEnv
    return _NEXT_STEP_ENV_CLS

    
class IsaacLabAdapter(VectorEnv):
    """Adapts Isaac Lab ``ManagerBasedRLEnv`` to the gymnasium ``VectorEnv`` API.

    Required by the gymnasium ``VectorWrapper`` chain (for example
    ``VectorNStepReward``). ``ManagerBasedRLEnv`` is vectorized but deliberately
    does *not* inherit from ``gymnasium.vector.VectorEnv``. It already exposes
    ``num_envs`` and goal-conditioned ``Dict`` observation spaces (one key per
    observation group), so this adapter forwards ``reset``/``step`` and the
    spaces.

    It also supplies the goal reward used by Hindsight Experience Replay. Isaac
    Lab manager-based envs compute rewards through their ``RewardManager`` and
    expose no goal-conditioned reward function, so HER (which resolves
    ``compute_reward`` via ``EnvWrapper.get_base_env``) needs one here. The
    online ``RewardManager`` term should match this sparse reward so collected
    and relabeled transitions share the same scale.
    """

    def __init__(self, env, distance_threshold: float | None = None):
        """Store the Isaac Lab env and the sparse-goal distance threshold.

        Args:
            env (Any): Isaac Lab ``ManagerBasedRLEnv`` (or NextStep subclass)
                to adapt.
            distance_threshold: Max L2 distance for a successful goal in
                ``compute_reward``. ``None`` leaves the sparse goal reward
                unconfigured, which makes ``compute_reward`` raise. HER
                relabeling recomputes rewards through that call, so it cannot
                run against this adapter without a real threshold; it fails on
                the first relabeled episode rather than at construction.
        """
        self._env = env
        self.num_envs = env.num_envs
        self.single_observation_space = env.single_observation_space
        self.single_action_space = env.single_action_space
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.distance_threshold = distance_threshold

    def reset(self, *, seed=None, options=None):
        """Reset the Isaac Lab env, forwarding ``seed`` only.

        Args:
            seed (Any): Seed forwarded to ``env.reset``.
            options (Any): Accepted for VectorEnv API compatibility; unused.

        Returns:
            reset_result (Any): Return value of the underlying ``env.reset``.
        """
        return self._env.reset(seed=seed)

    def step(self, action):
        """Step the Isaac Lab env with the given action batch.

        Args:
            action (Any): Action batch accepted by the underlying env.

        Returns:
            step_result (Any): Return value of the underlying ``env.step``.
        """
        return self._env.step(action)

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        """Sparse goal reward: 0 if within ``distance_threshold`` else -1 (batched).

        Args:
            achieved_goal (Any): Achieved goal array, shape ``(..., goal_dim)``.
            desired_goal (Any): Desired goal array, same trailing dim as
                ``achieved_goal``.
            info (Any): Unused; accepted for HER / gymnasium reward callables.

        Returns:
            reward (numpy.ndarray): Per-env sparse rewards as float32.

        Raises:
            ValueError: If ``distance_threshold`` is ``None``, i.e. never set
                via the ``distance_threshold`` argument to ``IsaacSimWrapper``
                / ``IsaacLabAdapter`` (``env.config.distance_threshold`` in a
                YAML config).
        """
        if self.distance_threshold is None:
            raise ValueError(
                "IsaacLabAdapter.compute_reward requires a numeric distance_threshold, "
                "but none was configured (it is None). Set the distance_threshold "
                "argument passed to IsaacSimWrapper / IsaacLabAdapter -- in a YAML "
                "config this is env.config.distance_threshold."
            )
        d = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
        return -(d > self.distance_threshold).astype(np.float32)

    def render(self, **kwargs):
        """Forward render kwargs to the Isaac Lab env.

        Args:
            **kwargs (Any): Keyword arguments passed to ``env.render``.

        Returns:
            frame (Any): Render output from the underlying env.
        """
        return self._env.render(**kwargs)

    def close(self):
        """Close the underlying Isaac Lab environment."""
        self._env.close()

    @property
    def spec(self):
        """Return the Isaac Lab env ``spec`` attribute if present.

        Returns:
            spec (Any): Env spec object, or ``None`` when absent.
        """
        return getattr(self._env, 'spec', None)


class IsaacSimWrapper(EnvWrapper):
    """Wrapper for Isaac Lab / Isaac Sim manager-based RL environments.

    Launches the Kit app when needed, builds a NextStep-mode
    ``ManagerBasedRLEnv`` from a ``module:Class`` config id, adapts it through
    ``IsaacLabAdapter``, and applies registered vector wrappers. Supports reset,
    step, serialization, and optional camera-enabled rendering.
    """

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
        distance_threshold:float|None=None,
        enable_cameras:bool=False,
    ):
        """Build an Isaac Sim env, adapt it, and optionally bound the action space.

        Args:
            cfg: Isaac Lab env config id as ``module.path:ConfigClassName``.
            num_envs: Number of parallel environments; written to
                ``cfg.scene.num_envs``.
            obs_key: Dict-observation key for the agent state (default
                ``"policy"``).
            goal_key: Dict key for the desired goal, or ``None`` if unused.
            ach_goal_key: Dict key for the achieved goal, or ``None`` if unused.
            wrappers: Optional list of wrapper specs from ``WRAPPER_REGISTRY``,
                each a dict with ``type`` and optional ``params``.
            render_mode: Kit launch mode; ``"headless"`` runs without a display
                window, any other value launches headed.
            seed: RNG seed stored on the env config; when ``None``, a random
                31-bit seed is drawn by the base class.
            distance_threshold: Sparse-goal success radius forwarded to
                ``IsaacLabAdapter``. ``None`` means the sparse goal reward is
                unconfigured; set a real value for HER runs on
                goal-conditioned Isaac envs.
            enable_cameras: Launch the Kit app with camera/tiled rendering
                enabled. Required for envs with camera sensors (multi-modal
                image observations); leave False for state-only envs (faster).
        """
        super().__init__(cfg, num_envs, obs_key, goal_key, ach_goal_key, wrappers, render_mode, seed)
        self.distance_threshold = distance_threshold
        self.enable_cameras = enable_cameras
        # Initialize env
        self.env = self._initialize_env()

        # Bound action space between [-1,1] if unbounded
        if isinstance(self.env.single_action_space, gym.spaces.Box):
            low = self.env.single_action_space.low
            high = self.env.single_action_space.high
            if np.isinf(low).any() or np.isinf(high).any():
                act_dim = self.env.single_action_space.shape[-1]
                space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
                self.env.single_action_space = space
                self.env.action_space = gym.vector.utils.batch_space(space, self.env.num_envs)


    def _initialize_env(self):
        """Create the Isaac Lab env, adapt it, and apply registered wrappers.

        Reuses an already-running Kit app when present; otherwise launches via
        ``AppLauncher``. Instantiates the config class from ``self.env_id``,
        builds ``NextStepManagerBasedRLEnv``, wraps it in ``IsaacLabAdapter``,
        then applies ``WRAPPER_REGISTRY`` entries from ``self.wrappers``.

        Returns:
            env (Any): Adapted Isaac Lab environment, possibly further wrapped.

        Raises:
            ModuleNotFoundError: If Isaac Lab / Isaac Sim packages cannot be
                imported and no Kit app is already running.
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
            app_launcher = AppLauncher(headless=(self.render_mode=='headless'), device="cuda:0", enable_cameras=self.enable_cameras)
            self.app = app_launcher.app

        # Lazy create NextStepManagerBasedRLEnv class
        NextStepManagerBasedRLEnv = _get_next_step_env_cls()

        module_path, class_name = self.env_id.split(':')
        cfg_class = getattr(importlib.import_module(module_path), class_name)
        cfg = cfg_class()
        cfg.scene.num_envs = self.num_envs
        cfg.sim.device = "cuda:0"
        cfg.seed = self.seed
        env = NextStepManagerBasedRLEnv(cfg=cfg)
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
        """Convert NumPy actions to a float32 tensor; pass tensors through.

        Args:
            actions: Action array or tensor to format.

        Returns:
            formatted (torch.Tensor | numpy.ndarray): Float32 tensor when
                ``actions`` was a NumPy array; otherwise ``actions`` unchanged.
        """
        if isinstance(actions, np.ndarray):
            return T.tensor(actions, dtype=T.float32)
        return actions

    @property
    def observation_space(self):
        """Get the observation space of the environment.

        Returns:
            space (gymnasium.spaces.Space): The vector observation space.
        """
        return self.env.observation_space
    
    @property
    def action_space(self):
        """Get the action space of the environment.

        Returns:
            space (gymnasium.spaces.Space): The vector action space.
        """
        return self.env.action_space

    @property
    def single_action_space(self):
        """Get the single-env action space for vectorized environments.

        Returns:
            space (gymnasium.spaces.Space): Action space of one sub-environment.
        """
        return self.env.single_action_space

    @property
    def single_observation_space(self):
        """Get the single-env observation space for vectorized environments.

        Returns:
            space (gymnasium.spaces.Space): Observation space of one
                sub-environment.
        """
        return self.env.single_observation_space
    
    def reset(self, seed:int|None=None):
        """Reset the vector env and return an ``Observation``.

        Args:
            seed: Seed for ``env.reset`` and the action space; defaults to
                ``self.seed`` when omitted.

        Returns:
            observation (Observation): Initial states, optional goals, and
                infos; may attach ``n_step_trajectory`` when present in infos.
        """
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
        """Close the underlying env and the Kit application."""
        self.env.close()
        self.app.close()

    def step(self, action)->Observation:
        """Step the vector env and pack the transition into an ``Observation``.

        Args:
            action (Any): Action batch accepted by the underlying vector env.

        Returns:
            Observation with states, rewards, terminations, truncations, and
                infos; may attach ``n_step_trajectory`` when present in infos.
        """
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
        """Build a JSON-serializable config with type ``isaacsim``.

        Returns:
            config (dict): Parent config with ``type`` set to ``"isaacsim"`` and
                ``distance_threshold`` / ``enable_cameras`` under nested
                ``config``.
        """
        config = super().config
        config['type'] = "isaacsim"
        config['config']['distance_threshold'] = self.distance_threshold
        config['config']['enable_cameras'] = self.enable_cameras
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
        """Serialize the wrapper configuration to JSON.

        Returns:
            json_string (str): JSON encoding of ``self.config``.
        """
        return json.dumps(self.config)

    @classmethod
    def from_json(cls, json_string):
        """Create an Isaac Sim wrapper instance from a JSON string.

        Args:
            json_string (str): JSON string with a nested ``config`` object of
                constructor kwargs.

        Returns:
            wrapper (IsaacSimWrapper): A new Isaac Sim wrapper instance.

        Raises:
            ValueError: If construction from the parsed config fails.
        """
        config = json.loads(json_string)
        config = config['config']
        try:
            return cls(**config)
        except Exception as e:
            raise ValueError(f"Environment wrapper error: {config}, {e}")


class CustomJSONEncoder(json.JSONEncoder):
    """JSON encoder for Gymnasium specs and ``GymnasiumWrapper`` configs."""

    def default(self, obj):
        """Serialize known env/wrapper objects; defer unknowns to the base.

        Args:
            obj (Any): Object ``json.dumps`` could not encode natively.

        Returns:
            value (dict | str): JSON-serializable form for ``EnvSpec``,
                ``WrapperSpec``, ``GymnasiumWrapper``, or callables; otherwise
                the base encoder raises ``TypeError``.
        """
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
    """Flatten a ``WrapperSpec`` (or stringify a non-spec) for JSON.

    Args:
        wrapper_spec (Any): Gymnasium ``WrapperSpec``, or any other object.

    Returns:
        result (dict | str): Attribute dict for a ``WrapperSpec`` (callables
            stringified), else ``str(wrapper_spec)``.
    """
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
    """Extract a JSON-friendly dict of common ``EnvSpec`` fields.

    Args:
        env_spec (EnvSpec): Gymnasium environment specification.

    Returns:
        spec_dict (dict): Common ``EnvSpec`` fields; ``additional_wrappers`` is
            always an empty list.
    """
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

