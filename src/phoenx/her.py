"""Hindsight Experience Replay (HER) — pluggable relabeler.

A buffer-agnostic strategy object that converts completed episodes into
hindsight-relabeled samples. Supports both off-policy buffers (ReplayBuffer /
PrioritizedReplayBuffer) which consume N-step windows, and on-policy buffers
(TrajectoryBuffer) which consume full trajectories.

==============================================================================
Output formats
==============================================================================

`output_format='n_step'` — for off-policy buffers
    Per-step goal sampling: for each step t in the original episode, sample k
    goals according to `strategy`. This yields M ≈ T_ep * k independent
    (state, action, goal) samples. Each becomes the start of an N-step window
    rebuilt from the original episode, with rewards recomputed under the
    relabeled goal.

    Returns a single Dict[str, Tensor] with leading dims (M, N, *feat) — the
    same layout VectorNStepReward._build_trajectories produces:
        repeat-padded: states, actions, next_states, *_achieved_goals, desired_goals
        zero-padded:   rewards, intrinsic_rewards, terminations, truncations
        trajectory_lengths: (M,) int64

`output_format='flat'` — for on-policy buffers
    Per-trajectory goal sampling: sample k goals once per episode; each goal
    relabels ALL T_ep steps. Yields k coherent relabeled trajectories that
    each look structurally identical to a freshly collected on-policy rollout
    — GAE works on them, IS ratios computed against current/old policy work,
    no special agent handling required.

    Returns List[Dict[str, Tensor]] with each dict's tensors of shape
    (T_ep, *feat). The caller extends `completed_trajectories` with it.

==============================================================================
Strategies (Andrychowicz et al. 2017)
==============================================================================

`'final'`   - Final achieved goal of the episode. k = 1 always.
              Available in both output formats.
`'future'`  - For each step t, sample k goals from achieved goals in [t, T_ep).
              Inherently per-step → only available in n_step mode.
`'episode'` - Sample k goals from achieved goals anywhere in the episode.
              In n_step mode: per-step; in flat mode: per-trajectory.
`'random'`  - Sample k goals from an external `AchievedGoalPool` of goals from
              past episodes. The pool is fed by the buffer on each completed
              episode (see ReplayBuffer / TrajectoryBuffer integration).

==============================================================================
Usage examples
==============================================================================

Off-policy with PRB (DDPG/TD3/SAC):
    relabeler = HindsightRelabeler(
        env, strategy='future', num_goals=4,
        output_format='n_step', N=N,
    )
    buffer = PrioritizedReplayBuffer(env, buffer_size=1_000_000, N=N,
                                     relabeler=relabeler)

On-policy with TrajectoryBuffer (PPO/A2C):
    relabeler = HindsightRelabeler(
        env, strategy='final',
        output_format='flat',
    )
    buffer = TrajectoryBuffer(env, buffer_size=2048, relabeler=relabeler)
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as T

from .env_wrapper import EnvWrapper
from .obs_utils import tree_index, tree_to
from .torch_utils import get_device


# ============================================================================
# AchievedGoalPool — backing storage for strategy='random'
# ============================================================================

class AchievedGoalPool:
    """FIFO ring buffer of achieved goals from past episodes.

    Required only by strategy='random'. The buffer's HER hook calls
    `pool.add(achieved_goals)` whenever an episode completes, populating the
    pool with goals that have been physically reached in some prior rollout.
    The relabeler then samples from it via `pool.sample(n)`.
    """

    def __init__(
        self,
        capacity: int,
        goal_dim: Tuple[int, ...],
        device: T.device,
    ):
        self.capacity = capacity
        self.device = device
        self.buffer = T.zeros((capacity, *goal_dim), dtype=T.float32, device=device)
        self.size = 0
        self.idx = 0

    def add(self, goals: T.Tensor) -> None:
        """Append a batch of achieved goals; wraps FIFO when full."""
        B = goals.shape[0]
        if B == 0:
            return
        goals = goals.to(self.device, dtype=T.float32)
        if B >= self.capacity:
            self.buffer.copy_(goals[-self.capacity:])
            self.size = self.capacity
            self.idx = 0
            return
        end = self.idx + B
        if end <= self.capacity:
            self.buffer[self.idx:end] = goals
        else:
            split = self.capacity - self.idx
            self.buffer[self.idx:] = goals[:split]
            self.buffer[: end % self.capacity] = goals[split:]
        self.idx = end % self.capacity
        self.size = min(self.size + B, self.capacity)

    def sample(self, n: int) -> T.Tensor:
        """Sample n goals uniformly with replacement. Caller must check size > 0."""
        idx = T.randint(0, self.size, (n,), device=self.device)
        return self.buffer[idx]


# ============================================================================
# HindsightRelabeler — the main strategy object
# ============================================================================

class HindsightRelabeler:
    """Buffer-agnostic HER relabeler. See module docstring for design overview."""

    _VALID_STRATEGIES = ("final", "future", "episode", "random")
    _VALID_OUTPUT_FORMATS = ("n_step", "flat")

    def __init__(
        self,
        env: EnvWrapper,
        strategy: str = "future",
        num_goals: int = 4,
        output_format: str = "n_step",
        N: int | None = None,
        gamma: float = 0.99,
        device: str | T.device | None = None,
        relabel_terminations: bool = False,
        future_lo: str = "inclusive",
        goal_pool: AchievedGoalPool | None = None,
    ):
        # --- Validate config -------------------------------------------------
        if strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy {strategy!r}. Must be one of {self._VALID_STRATEGIES}."
            )
        if output_format not in self._VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"Invalid output_format {output_format!r}. "
                f"Must be one of {self._VALID_OUTPUT_FORMATS}."
            )
        if output_format == "n_step" and (N is None or N < 1):
            raise ValueError("output_format='n_step' requires N >= 1.")
        if output_format == "flat" and strategy == "future":
            raise ValueError(
                "strategy='future' is inherently per-step and cannot be used with "
                "output_format='flat' (per-trajectory sampling). Use 'final', "
                "'episode', or 'random' instead."
            )
        if strategy == "random" and goal_pool is None:
            raise ValueError("strategy='random' requires a goal_pool argument.")
        if num_goals <= 0 and strategy != "final":
            raise ValueError(f"num_goals must be > 0 for strategy {strategy!r}.")
        if future_lo not in ("inclusive", "exclusive"):
            raise ValueError("future_lo must be 'inclusive' or 'exclusive'.")

        self.env = env
        self.strategy = strategy
        self.num_goals = num_goals
        self.output_format = output_format
        self.N = N
        self.gamma = gamma
        self.device = get_device(device)
        self.relabel_terminations = relabel_terminations
        self.future_lo = future_lo
        self.goal_pool = goal_pool

        # Cache the env's reward function once. Walks the wrapper stack to find
        # compute_reward on a sub-env (SyncVectorEnv doesn't expose it directly).
        self._compute_reward_fn = self._resolve_compute_reward(env)
        self._distance_threshold = self._resolve_distance_threshold(env)

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    @T.no_grad()
    def relabel_episode(
        self,
        episode: Dict[str, T.Tensor],
    ) -> Union[Dict[str, T.Tensor], List[Dict[str, T.Tensor]], None]:
        """Relabel one completed episode.

        Args:
            episode: Dict of (T_ep, *feat) tensors representing a single completed
                episode. Required keys: states, actions, rewards, next_states,
                terminations, truncations, state_achieved_goals,
                next_state_achieved_goals, desired_goals.
                Optional: intrinsic_rewards.

        Returns:
            - output_format='n_step': single Dict of (M, N, *feat) tensors ready
              for buffer.add(**result). None if no windows can be built.
            - output_format='flat': List[Dict] of (T_ep, *feat) trajectory dicts,
              one per relabeled copy. Empty list if no goals could be sampled.
        """
        ep_states = episode["states"]
        if isinstance(ep_states, dict):
            T_ep = int(next(iter(ep_states.values())).shape[0])
        else:
            T_ep = int(ep_states.shape[0])
        if T_ep == 0:
            return None if self.output_format == "n_step" else []

        if self.output_format == "n_step":
            return self._relabel_n_step(episode, T_ep)
        return self._relabel_flat(episode, T_ep)

    # ========================================================================
    # N-STEP PATH (off-policy buffers)
    # ========================================================================

    def _relabel_n_step(
        self,
        episode: Dict[str, T.Tensor],
        T_ep: int,
    ) -> Dict[str, T.Tensor] | None:
        """Per-step goal sampling → padded (M, N, *feat) windows dict."""
        starts, new_goals = self._sample_per_step_goals(episode, T_ep)
        if starts.numel() == 0:
            return None
        return self._build_nstep_windows(starts, new_goals, episode, T_ep)

    def _sample_per_step_goals(
        self,
        episode: Dict[str, T.Tensor],
        T_ep: int,
    ) -> Tuple[T.Tensor, T.Tensor]:
        """For each step t in the episode, sample k goals (or 1 for 'final').

        Returns:
            starts: (M,) int64 step indices into [0, T_ep) — where each window starts.
            new_goals: (M, *goal_dim) goals to use for the corresponding start.
        """
        next_ach = episode["next_state_achieved_goals"].to(self.device)
        goal_dim = next_ach.shape[1:]
        empty = (
            T.empty(0, dtype=T.long, device=self.device),
            T.empty((0, *goal_dim), dtype=T.float32, device=self.device),
        )

        if self.strategy == "final":
            # Every step relabeled with the final achieved goal — k = 1.
            starts = T.arange(T_ep, device=self.device)
            new_goals = (
                next_ach[T_ep - 1].unsqueeze(0).expand(T_ep, *goal_dim).contiguous()
            )
            return starts, new_goals

        k = self.num_goals

        if self.strategy == "future":
            # For each t, sample k goals from achieved goals in [t+offset, T_ep).
            # 'inclusive' (offset=0) matches SB3; 'exclusive' (offset=1) is the HER paper.
            offset = 0 if self.future_lo == "inclusive" else 1
            starts_list, goals_list = [], []
            for t in range(T_ep):
                lo = t + offset
                if lo >= T_ep:
                    continue  # No future steps available for t.
                idx = T.randint(lo, T_ep, (k,), device=self.device)
                starts_list.append(T.full((k,), t, dtype=T.long, device=self.device))
                goals_list.append(next_ach[idx])
            if not starts_list:
                return empty
            return T.cat(starts_list), T.cat(goals_list)

        if self.strategy == "episode":
            # For each t, sample k goals from anywhere in the episode.
            starts = T.arange(T_ep, device=self.device).repeat_interleave(k)
            idx = T.randint(0, T_ep, (T_ep * k,), device=self.device)
            return starts, next_ach[idx]

        # 'random'
        if self.goal_pool.size == 0:
            return empty  # Pool not yet populated — skip this episode.
        starts = T.arange(T_ep, device=self.device).repeat_interleave(k)
        new_goals = self.goal_pool.sample(T_ep * k).to(self.device)
        return starts, new_goals

    def _build_nstep_windows(
        self,
        starts: T.Tensor,      # (M,)
        new_goals: T.Tensor,   # (M, *goal_dim)
        episode: Dict[str, T.Tensor],
        T_ep: int,
    ) -> Dict[str, T.Tensor]:
        """Build (M, N, *feat) N-step windows matching VectorNStepReward's padding:
        state-like fields: repeat-padded (clamp out-of-range to last valid step)
        reward-like fields: zero-padded (gather + T.where mask)
        trajectory_lengths: count of valid steps per window
        """
        M = int(starts.shape[0])
        N = self.N
        dev = self.device
        ep = {
            k: (tree_to(v, device=dev) if isinstance(v, (T.Tensor, dict)) else v)
            for k, v in episode.items()
        }

        # Build the (M, N) index grid: window m's step n maps to original step starts[m]+n.
        offsets = T.arange(N, device=dev)
        ts = starts.unsqueeze(1) + offsets.unsqueeze(0)   # (M, N)
        valid = ts < T_ep                                 # (M, N) bool — in-range mask
        ts_repeat = T.clamp(ts, max=T_ep - 1)             # Repeat-pad indices.

        # Gather state-like fields (states may be dict-of-tensors, multi-modal).
        # Clamped indices give us repeat-padding for free.
        states          = tree_index(ep["states"], ts_repeat)
        actions         = ep["actions"][ts_repeat]
        next_states     = tree_index(ep["next_states"], ts_repeat)
        state_ach_goals = ep["state_achieved_goals"][ts_repeat]
        next_ach_goals  = ep["next_state_achieved_goals"][ts_repeat]

        # Each window has a single constant desired goal — broadcast across N.
        goal_shape = new_goals.shape[1:]
        desired_goals = new_goals.unsqueeze(1).expand(M, N, *goal_shape).contiguous()

        # Recompute rewards under the relabeled goals (single batched call).
        rewards = self._compute_reward_batched(
            next_ach_goals.reshape(-1, *goal_shape),
            desired_goals.reshape(-1, *goal_shape),
        ).view(M, N)
        rewards = T.where(valid, rewards, T.zeros_like(rewards))

        # Gather and zero-pad termination/truncation flags past episode end.
        terminations = ep["terminations"][ts_repeat]
        truncations  = ep["truncations"][ts_repeat]
        terminations = T.where(valid, terminations, T.zeros_like(terminations))
        truncations  = T.where(valid, truncations,  T.zeros_like(truncations))

        # Optional: auto-terminate at first achievement of relabeled goal.
        # OFF by default — HER paper leaves termination flags alone.
        if self.relabel_terminations:
            achieved = self._is_achieved(next_ach_goals, desired_goals)
            first_hit = achieved & (achieved.cumsum(dim=1) == 1) & valid
            terminations = terminations | first_hit

        # Pass through intrinsic rewards (curiosity / ICM) if present in the episode.
        if "intrinsic_rewards" in ep and ep["intrinsic_rewards"] is not None:
            intrinsic_rewards = ep["intrinsic_rewards"][ts_repeat]
            intrinsic_rewards = T.where(
                valid, intrinsic_rewards, T.zeros_like(intrinsic_rewards),
            )
        else:
            intrinsic_rewards = T.zeros((M, N), dtype=T.float32, device=dev)

        # Per-window count of valid (non-padded) steps.
        traj_lengths = T.minimum(
            T.full((M,), N, dtype=T.long, device=dev),
            (T_ep - starts).clamp(min=0),
        )

        return {
            "states":                    states,
            "actions":                   actions,
            "rewards":                   rewards,
            "intrinsic_rewards":         intrinsic_rewards,
            "next_states":               next_states,
            "terminations":              terminations,
            "truncations":               truncations,
            "state_achieved_goals":      state_ach_goals,
            "next_state_achieved_goals": next_ach_goals,
            "desired_goals":             desired_goals,
            "trajectory_lengths":        traj_lengths,
        }

    # ========================================================================
    # FLAT PATH (on-policy buffers)
    # ========================================================================

    def _relabel_flat(
        self,
        episode: Dict[str, T.Tensor],
        T_ep: int,
    ) -> List[Dict[str, T.Tensor]]:
        """Per-trajectory goal sampling → List[Dict] of relabeled trajectories."""
        new_goals = self._sample_per_trajectory_goals(episode, T_ep)  # (K, *goal_dim)
        if int(new_goals.shape[0]) == 0:
            return []
        return self._build_relabeled_trajectories(new_goals, episode, T_ep)

    def _sample_per_trajectory_goals(
        self,
        episode: Dict[str, T.Tensor],
        T_ep: int,
    ) -> T.Tensor:
        """Pick K goals total — one per relabeled copy.

        Returns: (K, *goal_dim) tensor. K = 1 for 'final', K = num_goals otherwise.
                 (0, *goal_dim) if no goals can be sampled (e.g. empty 'random' pool).
        """
        next_ach = episode["next_state_achieved_goals"].to(self.device)
        goal_dim = next_ach.shape[1:]

        if self.strategy == "final":
            return next_ach[T_ep - 1].unsqueeze(0)  # (1, *goal_dim)

        k = self.num_goals

        if self.strategy == "episode":
            # Sample k goal-indices from anywhere in the episode.
            idx = T.randint(0, T_ep, (k,), device=self.device)
            return next_ach[idx]

        if self.strategy == "random":
            if self.goal_pool.size == 0:
                return T.empty((0, *goal_dim), dtype=T.float32, device=self.device)
            return self.goal_pool.sample(k).to(self.device)

        # 'future' is rejected in __init__ for output_format='flat' — unreachable.
        raise RuntimeError(f"Unreachable strategy in flat mode: {self.strategy}")

    def _build_relabeled_trajectories(
        self,
        new_goals: T.Tensor,
        episode: Dict[str, T.Tensor],
        T_ep: int,
    ) -> List[Dict[str, T.Tensor]]:
        """Build K full trajectory dicts. Each dict shares its state/action arrays
        with the original episode (same shape (T_ep, *feat) as on-policy rollouts).
        Only desired_goals (broadcast new goal) and rewards (recomputed under it)
        differ — plus terminations if `relabel_terminations` is on.
        """
        K = int(new_goals.shape[0])
        dev = self.device
        ep = {
            k: (tree_to(v, device=dev) if isinstance(v, (T.Tensor, dict)) else v)
            for k, v in episode.items()
        }
        goal_shape = new_goals.shape[1:]
        next_ach = ep["next_state_achieved_goals"]
        has_ir = "intrinsic_rewards" in ep and ep["intrinsic_rewards"] is not None

        trajectories: List[Dict[str, T.Tensor]] = []
        for j in range(K):
            # Broadcast this copy's single goal across all T_ep steps.
            desired_goals = (
                new_goals[j].unsqueeze(0).expand(T_ep, *goal_shape).contiguous()
            )

            # Recompute rewards for the whole trajectory under the new goal.
            rewards = self._compute_reward_batched(next_ach, desired_goals)

            terminations = ep["terminations"].clone()
            if self.relabel_terminations:
                achieved = self._is_achieved(next_ach, desired_goals)
                first_hit = achieved & (achieved.cumsum(dim=0) == 1)
                terminations = terminations | first_hit

            traj = {
                "states":                    ep["states"],
                "actions":                   ep["actions"],
                "rewards":                   rewards,
                "next_states":               ep["next_states"],
                "terminations":              terminations,
                "truncations":               ep["truncations"].clone(),
                "state_achieved_goals":      ep["state_achieved_goals"],
                "next_state_achieved_goals": next_ach,
                "desired_goals":             desired_goals,
            }
            if has_ir:
                traj["intrinsic_rewards"] = ep["intrinsic_rewards"]
            trajectories.append(traj)

        return trajectories

    # ========================================================================
    # ENV HOOKS — locate compute_reward / distance_threshold on the env stack
    # ========================================================================

    @staticmethod
    def _resolve_compute_reward(env: EnvWrapper):
        """Walks EnvWrapper.env → ... → SyncVectorEnv → envs[0].unwrapped looking
        for compute_reward(achieved, desired, info). Falls back to
        env.get_base_env() if that yields something exposing it directly
        (e.g. IsaacSim envs).
        """
        try:
            base = env.get_base_env() if hasattr(env, "get_base_env") else None
        except Exception:
            base = None
        if base is not None and hasattr(base, "compute_reward"):
            return base.compute_reward

        # SyncVectorEnv doesn't expose compute_reward — drill into its first sub-env.
        cur = getattr(env, "env", None)
        while cur is not None:
            sub_envs = getattr(cur, "envs", None)
            if sub_envs:
                sub = sub_envs[0]
                core = sub.unwrapped if isinstance(sub, gym.Env) else sub
                if hasattr(core, "compute_reward"):
                    return core.compute_reward
                break  # Found the vec layer; nothing else worth checking.
            cur = getattr(cur, "env", None)

        raise AttributeError(
            "HindsightRelabeler could not locate compute_reward on the env stack. "
            "HER requires a goal-conditioned env (e.g. gymnasium-robotics Fetch* / "
            "AntMaze*) that exposes compute_reward(achieved, desired, info)."
        )

    @staticmethod
    def _resolve_distance_threshold(env: EnvWrapper) -> float | None:
        """Best-effort lookup of env.distance_threshold (only used by relabel_terminations=True)."""
        try:
            base = env.get_base_env() if hasattr(env, "get_base_env") else None
        except Exception:
            base = None
        if base is not None and hasattr(base, "distance_threshold"):
            return float(base.distance_threshold)
        cur = getattr(env, "env", None)
        while cur is not None:
            sub_envs = getattr(cur, "envs", None)
            if sub_envs:
                sub = sub_envs[0]
                core = sub.unwrapped if isinstance(sub, gym.Env) else sub
                if hasattr(core, "distance_threshold"):
                    return float(core.distance_threshold)
                break
            cur = getattr(cur, "env", None)
        return None

    def _compute_reward_batched(
        self,
        achieved: T.Tensor,
        desired: T.Tensor,
    ) -> T.Tensor:
        """Batched compute_reward call with per-row fallback for non-broadcasting envs."""
        ach_np = achieved.detach().cpu().numpy()
        des_np = desired.detach().cpu().numpy()
        try:
            r = self._compute_reward_fn(ach_np, des_np, {})
            r = np.asarray(r, dtype=np.float32).reshape(-1)
            if r.shape[0] != ach_np.shape[0]:
                raise ValueError("compute_reward returned wrong batch shape; falling back.")
        except Exception:
            r = np.fromiter(
                (self._compute_reward_fn(ach_np[i], des_np[i], {})
                 for i in range(ach_np.shape[0])),
                dtype=np.float32,
                count=ach_np.shape[0],
            )
        return T.tensor(r, dtype=T.float32, device=self.device)

    def _is_achieved(self, ach: T.Tensor, desired: T.Tensor) -> T.Tensor:
        """Boolean achievement check; used only when relabel_terminations=True."""
        if self._distance_threshold is not None:
            return T.linalg.norm(ach - desired, dim=-1) <= self._distance_threshold
        # Fallback: assume sparse {-1, 0} reward; achieved iff reward >= 0.
        leading = ach.shape[:-1]
        flat_ach = ach.reshape(-1, ach.shape[-1])
        flat_des = desired.reshape(-1, desired.shape[-1])
        r = self._compute_reward_batched(flat_ach, flat_des)
        return (r >= 0.0).reshape(leading)

    # ========================================================================
    # MISC
    # ========================================================================

    def get_config(self) -> Dict[str, Any]:
        """Serializable config dict for logging / experiment reproducibility."""
        return {
            "type": self.__class__.__name__,
            "strategy": self.strategy,
            "num_goals": self.num_goals,
            "output_format": self.output_format,
            "N": self.N,
            "gamma": self.gamma,
            "device": str(self.device),
            "relabel_terminations": self.relabel_terminations,
            "future_lo": self.future_lo,
            "uses_goal_pool": self.goal_pool is not None,
        }
