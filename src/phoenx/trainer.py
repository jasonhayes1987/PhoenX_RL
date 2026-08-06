"""Train and evaluate agents against Gymnasium / Isaac environments.

``Trainer`` owns the outer loop: reset, step, optional learn, logging, and
checkpointing. ``TrainingSchedule`` decides how long to run and when to call
``learn``; ``SuccessCriterion`` turns episode outcomes into a success flag for
logging. Together they are the object graph that
[build_trainer_from_config][phoenx.builder.build_trainer_from_config] assembles
from YAML.
"""

from collections import deque
import json
import random
from pathlib import Path
import time
from datetime import timedelta
from dataclasses import dataclass, replace
from typing import Literal, Any

from rich.live import Live
from rich.table import Table
from rich.console import Console
import numpy as np
import torch as T

from .torch_utils import set_seed
from .rl_callbacks import Callback, WandbCallback, load as build_callback
from .rl_agents import Agent, HasTargetNetworks, build_agent
from .env_wrapper import EnvWrapper, Observation
from .obs_utils import flatten_obs, tree_index, tree_to
from .buffer import Buffer, PrioritizedReplayBuffer
from .her import HindsightRelabeler
from .renderer import Renderer
from .logging_config import get_logger

@dataclass
class TrainingSchedule:
    """When to stop a run and how often the agent should learn.

    Attributes:
        stop_unit: Whether ``stop_units`` counts timesteps or episodes.
        stop_units: Training length in ``stop_unit`` units.
        learn_every_unit: Whether ``learn_every`` counts timesteps or episodes.
        learn_every: Progress between consecutive ``learn`` calls.
        updates_per_learn: Gradient updates performed per ``learn`` call.
        batch_size: Transitions drawn per update when ``updates_per_learn > 1``.
        mini_batch_size: Mini-batch size forwarded to ``agent.learn``.
        learning_epochs: Epochs forwarded to ``agent.learn``.
        warmup_steps: Environment steps collected before the first learn.
        seed: Optional RNG seed; ``None`` draws a random seed at run start.
    """
    # Training Length
    stop_unit: Literal["timestep", "episode"] = "timestep" # Train length unit
    stop_units: int = 1_000_000 # Train length
    # Learn Frequency
    learn_every_unit: Literal["timestep", "episode"] = "timestep" # Learn frequency unit
    learn_every: int = 1 # Learn frequency
    # Per learn() call
    updates_per_learn: int = 1 # Number of updates per learn() call
    batch_size: int = 1 # Batch size
    mini_batch_size: int = 1 # Mini batch size
    learning_epochs: int = 1 # Number of learning epochs
    # Misc.
    warmup_steps: int = 0 # Optional warmup gate (no learning until this many steps)
    seed: int | None = None # Seed

    def is_done(self, *, step: int, episodes: int) -> bool:
        """Return whether the run has reached its configured length.

        Args:
            step: Cumulative environment steps so far.
            episodes: Cumulative completed episodes so far.

        Returns:
            ``True`` when progress in ``stop_unit`` meets ``stop_units``.
        """
        if self.stop_unit == "timestep":
            return step >= self.stop_units
        return episodes >= self.stop_units
    
    def should_learn(
        self, *,
        step: int,
        episodes: int,
        last_learn_at: int,
    ) -> bool:
        """Return whether it is time for another ``learn`` call.

        Args:
            step: Cumulative environment steps so far.
            episodes: Cumulative completed episodes so far.
            last_learn_at: Progress (in ``learn_every_unit``) at the previous
                learn gate.

        Returns:
            ``False`` during warmup; otherwise ``True`` when progress has
                advanced by at least ``learn_every`` since ``last_learn_at``.
        """
        if step < self.warmup_steps:
            return False
        progress = {
            "timestep": step,
            "episode": episodes,
        }[self.learn_every_unit]
        return progress >= last_learn_at + self.learn_every

    def get_config(self) -> dict:
        """Return a JSON-safe dict of the schedule fields.

        Returns:
            Mapping of constructor field names to their current values.
        """
        return {
            "stop_unit": self.stop_unit,
            "stop_units": self.stop_units,
            "learn_every_unit": self.learn_every_unit,
            "learn_every": self.learn_every,
            "updates_per_learn": self.updates_per_learn,
            "batch_size": self.batch_size,
            "mini_batch_size": self.mini_batch_size,
            "learning_epochs": self.learning_epochs,
            "warmup_steps": self.warmup_steps,
            "seed": self.seed,
        }


@dataclass
class SuccessCriterion:
    """Defines what counts as a successful episode.

    Attributes:
        metric: How success is decided. ``"info_flag"`` reads an env info key
            (default ``is_success``); ``"goal_distance"`` checks
            ``||achieved - desired|| <= threshold``; ``"episode_reward"``
            checks ``episode_reward >= threshold``.
        threshold: Cutoff required for ``goal_distance`` and
            ``episode_reward``; unused for ``info_flag``.
        info_key: Info-dict key used when ``metric="info_flag"``.
    """
    metric: Literal["info_flag", "goal_distance", "episode_reward"]
    threshold: float | None = None
    info_key: str = "is_success"

    def __post_init__(self):
        """Validate that threshold-dependent metrics supply a threshold.

        Raises:
            ValueError: If ``metric`` is ``goal_distance`` or
                ``episode_reward`` and ``threshold`` is ``None``.
        """
        if self.metric in ("goal_distance", "episode_reward") and self.threshold is None:
            raise ValueError(f"metric '{self.metric}' requires a threshold")

    def evaluate(self, obs: Observation, env_idx: int, episode_reward: float) -> bool | None:
        """Evaluate success for one finished environment index.

        Args:
            obs: Terminal observation for the finished episode.
            env_idx: Parallel-env index that just completed.
            episode_reward: Cumulative extrinsic reward for that episode.

        Returns:
            ``True`` / ``False`` when the metric can be computed, or ``None``
                when the required fields are missing from ``obs``.
        """
        if self.metric == "episode_reward":
            return episode_reward >= self.threshold
        if self.metric == "goal_distance":
            if obs.goals is None or obs.ach_goals is None:
                return None
            distance = T.norm(obs.ach_goals[env_idx] - obs.goals[env_idx], p=2, dim=-1)
            return bool(distance <= self.threshold)
        # info_flag
        flag = (obs.infos or {}).get(self.info_key)
        if flag is None:
            return None
        return bool(flag[env_idx]) if hasattr(flag, "__getitem__") else bool(flag)

    def get_config(self) -> dict:
        """Return a JSON-safe dict of the criterion fields.

        Returns:
            Mapping of constructor field names to their current values.
        """
        return {
            "metric": self.metric,
            "threshold": self.threshold,
            "info_key": self.info_key,
        }

class Trainer:
    """Outer training / evaluation loop for a wired agent, env, and schedule.

    Steps the environment, records into the buffer, calls ``agent.learn`` on the
    schedule's cadence, drives callbacks and optional rendering, and checkpoints
    when the running average reward improves.
    """

    def __init__(
        self,
        agent:Agent,
        env:EnvWrapper,
        schedule:TrainingSchedule,
        success_criterion:SuccessCriterion|None = None,
        buffer:Buffer|None = None,
        renderer:Renderer|None = None,
        callbacks:list[Callback]|None = None,
        log_level: str = 'INFO',
        save_dir: str = 'models/',
    ):
        """Wire the agent, environment, schedule, and optional run services.

        Args:
            agent: Policy / critic stack that acts and learns.
            env: Vectorized environment wrapper to step.
            schedule: Stop length, learn cadence, and batch sizes.
            success_criterion: Optional episode-success evaluator for logging.
            buffer: Replay or rollout buffer used during training.
            renderer: Optional episode renderer / video writer.
            callbacks: Optional list of train/test lifecycle callbacks.
            log_level: Logger level name (e.g. ``"INFO"``).
            save_dir: Directory for checkpoints and renderer output.
        """
        self.agent = agent
        self.env = env
        self.schedule = schedule
        self.success_criterion = success_criterion
        self.buffer = buffer
        self.renderer = renderer
        self.callbacks = callbacks
        self.log_level = log_level
        self.save_dir = save_dir

        # Set Renderer save dir (the agent owns its own save_dir via its config)
        if self.renderer is not None:
            self.renderer.save_dir = self.save_dir

        # Initialize internal attributes
        self.logger = get_logger(self.__class__.__name__, level=log_level.upper())
        self._initialized = False
        self._wandb = False
        self._step = None
        self._prev_obs = None
        self._prev_done = None
        self._best_reward = None
        self._episode_steps = None
        self._completed_episodes = None
        self._episode_scores = None
        self._score_history = None
        self._last_learn = None
        self._success_tracker = None
        # Checkpoint snapshots applied by _initialize_run when resuming (Trainer.load).
        self._resume_state = None
        self._resume_rng = None

    def _initialize_callbacks(self):
        """Initialize and configure callbacks for logging and monitoring."""
        try:
            if self.callbacks:
                for callback in self.callbacks:
                    callback._config(self.get_config())
                    if isinstance(callback, WandbCallback):
                        self._wandb = True

        except Exception as e:
            raise ValueError(f"Error initializing callbacks: {e}")

    def _initialize_run(self, context: Literal["train", "test"], **kwargs: Any):
        """Initialize env, seed, model modes, and run-tracking counters.

        Args:
            context: ``"train"`` or ``"test"``; selects model/normalizer mode
                and which callback begin hooks fire.
            **kwargs: Extra keys merged into the config passed to callbacks.
        """
        if self._initialized:
            return
        
        # Set the composite model to train mode if training, else evaluation mode
        model = getattr(self.agent, 'model', None)
        if model is not None:
            if context == "train":
                model.train()
                model.logger.debug("Set model to train mode")
            elif context == "test":
                model.eval()
                model.logger.debug("Set model to eval mode")

        # Set target model to eval mode
        target_model = getattr(self.agent, 'target_model', None)
        if target_model is not None:
            target_model.eval()
            target_model.logger.debug("Set target_model to eval mode")

        # Fresh recurrent state for the new run
        if hasattr(self.agent, 'reset_hidden'):
            self.agent.reset_hidden()

        # Set VectorNStepReward wrapper Intrinsic Motivation pointer
        im = getattr(self.agent, 'intrinsic_motivation', None)
        if im is not None:
            nstep_wrapper = self.env._find_nstep_wrapper()
            if nstep_wrapper is not None:
                nstep_wrapper.set_intrinsic_motivation(im)

        # Set normalizers to train or eval mode
        self.set_normalizers(context)

        # Set internal attributes
        seed = self.schedule.seed if self.schedule.seed else T.randint(2**31-1, (1,)).item()
        set_seed(seed)
        observation = self.env.reset(seed=seed)

        # Set callbacks
        if self.callbacks:
            # self._initialize_callbacks()
            config = self.get_config()
            # config.update({'num_envs': self.env.num_envs, 'seed': seed})
            config.update(kwargs)

            models = [model for model in [getattr(self.agent, "model", None),
                                          self.agent.curiosity if hasattr(self.agent, 'curiosity') else None] if model is not None]
            
            func_name = 'on_train_begin' if context == "train" else 'on_test_begin'

            for callback in self.callbacks:
                callback_func = getattr(callback, func_name)
                if isinstance(callback, WandbCallback):
                    run_number = callback.run_name.split("-")[-1] if callback.run_name else None
                    if func_name == 'on_train_begin':
                        callback_func(config, run_number, tuple(models))
                    else:
                        callback_func(config, run_number)
                else:
                    callback_func(config)
        self._initialized = True

        # Set internal attributes
        self._step = 0
        self._prev_obs = observation
        self._best_reward = -T.inf
        self._episode_steps = T.zeros(self.env.num_envs, dtype=T.int32, device=self.agent.device)
        self._completed_episodes = T.zeros(self.env.num_envs, dtype=T.int32, device=self.agent.device)
        self._episode_scores = T.zeros(self.env.num_envs, dtype=T.float32, device=self.agent.device)
        self._prev_done = T.zeros(self.env.num_envs, dtype=T.bool, device=self.agent.device)
        self._score_history = deque(maxlen=100)
        self._last_learn = 0
        self._success_tracker = deque(maxlen=100)

        # If resuming (Trainer.load), overlay the checkpointed counters + RNG.
        self._apply_resume_state(context)

    def _apply_resume_state(self, context: Literal["train", "test"]) -> None:
        """Overlay checkpointed counters and RNG onto a freshly initialized run.

        Applies only when resuming *training* so the run continues from the
        saved step/episode progress and RNG stream. During evaluation the fresh
        zeroed counters are kept (the test loop's stop condition depends on
        them), and any staged snapshots are discarded.
        """
        if context != "train":
            self._resume_state = None
            self._resume_rng = None
            return

        state = self._resume_state
        if state is not None:
            self._step = state.get("_step", self._step)
            self._best_reward = state.get("_best_reward", self._best_reward)
            for name in ("_completed_episodes", "_episode_scores", "_episode_steps"):
                value = state.get(name)
                if value is not None:
                    setattr(self, name, value.to(self.agent.device))
            if state.get("_score_history") is not None:
                self._score_history = deque(state["_score_history"], maxlen=100)
            if state.get("_success_tracker") is not None:
                self._success_tracker = deque(state["_success_tracker"], maxlen=100)
            if getattr(self, "_resume_buffer_restored", False):
                self._last_learn = state.get("_last_learn", self._last_learn)
            else:
                # Start with an empty buffer. Re-anchor the learn gate so a full window
                # fresh data is collected before the first learn.
                self._last_learn = {
                    "timestep": self._step,
                    "episode": self._completed_episodes.sum().item(),
                }[self.schedule.learn_every_unit]
            self._resume_state = None

        rng = self._resume_rng
        if rng is not None:
            try:
                T.set_rng_state(rng["torch"])
                np.random.set_state(rng["numpy"])
                random.setstate(rng["python"])
                if rng.get("cuda") is not None and T.cuda.is_available():
                    T.cuda.set_rng_state_all(rng["cuda"])
            except Exception as e:
                self.logger.warning(f"Could not fully restore RNG state on resume: {e}")
            self._resume_rng = None

    # def _find_nstep_wrapper(self, env: EnvWrapper) -> VectorNStepReward | None:
    #     """Finds the VectorNStepReward wrapper in the environment chain."""
    #     while env is not None:
    #         if isinstance(env, VectorNStepReward):
    #             return env
    #         env = getattr(env, 'env', None)

    def _iter_normalizers(self):
        """Yield ``(name, normalizer)`` for every normalizer the agent has."""
        for name in ("state_normalizer", "goal_normalizer",
                    "reward_normalizer", "advantage_normalizer"):
            norm = getattr(self.agent, name, None)
            if norm is not None:
                yield name, norm

    def _apply_per_update(self, sample: dict, learn_metrics: dict) -> None:
        """Push TD errors into a prioritized buffer and collect PER metrics.

        No-op unless ``self.buffer`` is a ``PrioritizedReplayBuffer``. Pops
        ``td_errors`` from ``learn_metrics`` so a per-sample tensor never
        reaches scalar loggers.

        Args:
            sample: Batch returned by ``buffer.sample``, including ``indices``.
            learn_metrics: Metrics dict from ``agent.learn``; mutated in place.
        """
        # Pop so a per-sample tensor never leaks into wandb scalar logs.
        td_errors = learn_metrics.pop("td_errors", None)

        if not isinstance(self.buffer, PrioritizedReplayBuffer):
            return
        indices = sample.get("indices")
        if indices is None or td_errors is None:
            return

        self.buffer.update_priorities(
            indices,
            td_errors.detach().flatten().to(self.buffer.device),
        )

        # if self._wandb:
        learn_metrics.update(self._collect_per_metrics(sample, indices))

    def _collect_per_metrics(self, sample: dict, indices: T.Tensor) -> dict:
        """Build scalar PER diagnostics for the latest sampled indices.

        Args:
            sample: Sampled batch carrying ``weights`` and ``probs``.
            indices: Buffer indices of the samples just trained on.

        Returns:
            Mapping of ``PER/...`` scalar metric names to float values.
        """
        # pb = self.buffer  # PrioritizedReplayBuffer
        actual_size = min(self.buffer.samples_added, self.buffer.buffer_size)
        valid = T.arange(actual_size, device=self.buffer.device)

        if self.buffer.priority == "proportional":
            offset = self.buffer.sum_tree.capacity - 1
            sampled_pri = self.buffer.sum_tree.tree[indices + offset]
            buffer_pri = self.buffer.sum_tree.tree[valid + offset]
        else:  # rank
            sampled_pri = self.buffer.priorities[indices]
            buffer_pri = self.buffer.priorities[valid]

        weights = sample["weights"]
        probs = sample["probs"]
        return {
            "PER/beta": self.buffer.beta,
            "PER/sampled_priority_mean": sampled_pri.mean().item(),
            "PER/sampled_priority_max": sampled_pri.max().item(),
            "PER/buffer_priority_mean": buffer_pri.mean().item(),
            "PER/buffer_priority_max": buffer_pri.max().item(),
            "PER/weight_mean": weights.mean().item(),
            "PER/weight_std": weights.std().item(),
            "PER/prob_mean": probs.mean().item(),
            "PER/prob_max": probs.max().item(),
        }

    def set_normalizers(self, context: Literal["train", "test"]):
        """Set every agent normalizer to train or eval mode.

        Args:
            context: ``"train"`` calls ``norm.train()``; ``"test"`` calls
                ``norm.eval()``.

        Raises:
            ValueError: If ``context`` is not ``"train"`` or ``"test"``.
        """
        if context not in ("train", "test"):
            raise ValueError(f"Invalid context: {context}")
        for _, norm in self._iter_normalizers():
            norm.train() if context == "train" else norm.eval()
        
        # # Set Intrinsic Motivation normalizers if present
        # im = getattr(self.agent, 'intrinsic_motivation', None)
        # if im is not None:
        #     im.set_normalizers_mode(context)

    def add_to_normalizers(self, obs: Observation):
        """Feed relevant fields from ``obs`` into the agent normalizers.

        Args:
            obs: Observation whose states, goals, and rewards are accumulated.
        """
        for name, norm in self._iter_normalizers():
            if name == "state_normalizer":
                norm.add(tree_to(obs.states, device=norm.device))

            elif name == "goal_normalizer" and obs.goals is not None:
                goals = T.cat([obs.goals, obs.ach_goals], dim=0).to(device=norm.device)
                norm.add(goals)

            elif name == "reward_normalizer":
                dones = T.logical_or(obs.terminations, obs.truncations)
                norm.add(obs.rewards, dones)

        # # Pass obs to Intrinsic Motivation if present
        # im = getattr(self.agent, 'intrinsic_motivation', None)
        # if im is not None:
        #     im.add_to_normalizers(obs)

    def update_normalizers(self):
        """Recompute running statistics on every agent normalizer."""
        for _, norm in self._iter_normalizers():
            norm.update()

        # # Update Intrinsic Motivation normalizers if present
        # im = getattr(self.agent, 'intrinsic_motivation', None)
        # if im is not None:
        #     im.update_normalizers()

    def normalize_observation(
        self, obs: Observation)->Observation:
        """Return a copy of ``obs`` with normalized states, goals, and rewards.

        Args:
            obs: Observation to normalize.

        Returns:
            New ``Observation`` with normalized fields; other fields unchanged.
        """
        states = obs.states
        goals = obs.goals
        ach_goals = obs.ach_goals
        rewards = obs.rewards
        for name, norm in self._iter_normalizers():
            if name == "state_normalizer":
                states = norm.normalize(obs.states)
            elif name == "goal_normalizer" and obs.goals is not None:
                goals = norm.normalize(obs.goals)
                ach_goals = norm.normalize(obs.ach_goals)
            elif name == "reward_normalizer" and obs.rewards is not None:
                rewards = norm.normalize(obs.rewards)
            
        return replace(obs, states=states, goals=goals, ach_goals=ach_goals, rewards=rewards)

    def update_schedulers(self):
        """Step every attached schedule by ``env.num_envs``.

        Covers agent entropy / noise / temperature schedules, intrinsic-reward
        schedulers (including composite components), and per-module LR
        schedulers on the composite model.
        """
        schedulers = [
            getattr(self.agent, name, None)
            for name in ("entropy_schedule", "noise_schedule", "target_noise_schedule")
        ] + [
            getattr(getattr(self.agent, "policy", None), "temperature_schedule", None)
        ] + [
            getattr(getattr(self.agent, "intrinsic_motivation", None), "reward_scheduler", None)
        ]
        # Per-module LR schedulers live on the composite model.
        model = getattr(self.agent, "model", None)
        if model is not None:
            schedulers.extend(getattr(model, "lr_schedulers", {}).values())
        # If using CompositeIntrinsicMotivation object, grab reward schedulers from each component
        # in the composite if any and update
        components = getattr(getattr(self.agent, 'intrinsic_motivation', None), 'components', None)
        if components is not None:
            for c in components:
                schedulers.append(getattr(c, 'reward_scheduler', None))

        for s in schedulers:
            if s is not None:
                s.step(self.env.num_envs)

    def step(self, training: bool = True):
        """Advance every parallel env by one step; optionally record to buffer.

        Selects an action, steps the env, updates episode trackers, and builds
        per-episode logs when any env finishes. During training, transitions
        are written to the buffer and a best-checkpoint save may fire.

        Args:
            training: When ``True``, record into the buffer and allow
                best-model checkpointing; when ``False``, evaluation only.

        Returns:
            result (dict): Mapping with ``step_log`` (per-step scalars) and
                ``episode_logs`` (one dict per finished env this step).
        """
        step_log = {}
        episode_logs = []

        # Normalize observations and goals if normalizers
        obs_norm = self.normalize_observation(self._prev_obs)
        action = self.get_action(obs_norm.states, obs_norm.goals, context='train' if training else 'test')
        # If using n-step, feed VectorNStepReward wrapper Action data
        nstep_wrapper = self.env._find_nstep_wrapper()
        if nstep_wrapper is not None:
            nstep_wrapper.set_action(action)
        # Take action in environment and get new Observation
        observation = self.env.step(action.actions)
        # If Agent uses Intrinsic Motivation, calculate intrinsic rewards to store in buffer (Training Only)
        if training:
            im = getattr(self.agent, 'intrinsic_motivation', None)
            if im is not None:
                # Pass normalized (flattened for IM) states to IM if present
                next_obs_norm = self.normalize_observation(observation)
                intrinsic_rewards = im.compute_rollout_reward(flatten_obs(obs_norm.states), flatten_obs(next_obs_norm.states), action.actions, env_indices = T.arange(self.env.num_envs, device=im.device))
            else:
                intrinsic_rewards = T.zeros_like(observation.rewards)
        else:
            intrinsic_rewards = T.zeros_like(observation.rewards)
        observation = replace(observation, intrinsic_rewards=intrinsic_rewards)

        dones = T.logical_or(observation.terminations, observation.truncations)
        valid_steps = ~self._prev_done
       
        
        # Add transitions to the buffer (non-normalized) (Training Only)
        if training:
            self.buffer.record(observation, prev_observation = self._prev_obs, actions = action, prev_dones = self._prev_done)

        # Increment episode steps and rewards
        self._episode_steps[valid_steps] += 1
        self._episode_scores[valid_steps] += observation.rewards[valid_steps].flatten()

        # Add step metrics to step log
        step_log.update({
            'step_reward': observation.rewards.mean().item(),
            'step_intrinsic_reward': observation.intrinsic_rewards.mean().item() if self.agent.intrinsic_motivation else 0.0
        })
        
        # Check if any env is done
        done_episodes = T.logical_or(observation.terminations, observation.truncations).nonzero(as_tuple=False).flatten()

        for i in done_episodes:
            self._completed_episodes[i] += 1
            self._score_history.append(float(self._episode_scores[i].item()))
            avg_reward = sum(self._score_history) / len(self._score_history)
            # Create episode log
            episode_log = {
                'env': i,
                'episode': int(self._completed_episodes.sum()),
                'episode_steps': int(self._episode_steps[i].item()),
                'episode_reward': round(float(self._episode_scores[i]), 2),
                'avg_reward': round(float(avg_reward), 2)
            }
            # Success metric
            if self.success_criterion is not None:
                success = self.success_criterion.evaluate(observation, i, float(self._episode_scores[i]))
                if success is not None:
                    self._success_tracker.append(int(success))
                    episode_log['success_rate'] = round(
                        sum(self._success_tracker) / len(self._success_tracker), 3
                    )
            # Goal distance tracking
            if self.env.goal_key is not None:
                goal_distance = T.norm(observation.ach_goals[i] - observation.goals[i], p=2, dim=-1)
                episode_log['goal_distance'] = round(goal_distance.item(), 3)
            episode_logs.append(episode_log)

        if done_episodes.numel() > 0 and training:
            if avg_reward > self._best_reward:
                self._best_reward = avg_reward
                episode_logs[-1]['best'] = True
                self.save()


        # set _cur_obs to observation
        self._prev_done = dones.clone()
        self._prev_obs = observation

        return{
        'step_log': step_log,
        'episode_logs': episode_logs,
    }

    def get_action(self,
        states: np.ndarray|T.Tensor,
        goals: np.ndarray|T.Tensor|None=None,
        context: str = 'train',
    )->T.Tensor:
        """Select an action from the current policy.

        Args:
            states: Current states (possibly already normalized).
            goals: Optional goal tensor for goal-conditioned policies.
            context: ``"train"`` or ``"test"`` forwarded to ``agent.act``.

        Returns:
            Value returned by ``agent.act`` (typically an ``Action`` dataclass;
                the signature annotates ``T.Tensor``).
        """
        return self.agent.act(
            states,
            goals,
            context,
            step = self._step,
            warmup = self.schedule.warmup_steps,
            dones = self._prev_done,  # recurrent agents reset hidden per env
        )

    @staticmethod
    def _shuffle_sample(sample: dict) -> dict:
        """Apply one random permutation across every tensor in a sampled batch.

        Dict-of-tensors values are permuted leaf-wise with the same index order
        so batch alignment is preserved.

        Args:
            sample: Batch mapping whose leading dim is the batch axis.

        Returns:
            New mapping with every tensor (or dict-of-tensors) permuted alike.
        """
        ref = sample.get("states")
        if ref is None:
            return sample
        if isinstance(ref, dict):
            ref = next(iter(ref.values()))
        perm = T.randperm(ref.shape[0], device=ref.device)

        def _permute(v):
            if isinstance(v, T.Tensor):
                return v[perm.to(v.device)]
            if isinstance(v, dict):
                return {k: leaf[perm.to(leaf.device)] for k, leaf in v.items()}
            return v

        return {k: _permute(v) for k, v in sample.items()}

    def learn(self)->dict:
        """Sample from the buffer and call ``agent.learn`` per the schedule.

        Draws ``updates_per_learn * batch_size`` transitions when the buffer
        is ready. With a single update the whole sample is passed through;
        with multiple updates the sample is shuffled and sliced into
        ``batch_size`` chunks. Each update runs
        ``_apply_per_update`` for prioritized replay.

        Returns:
            Metrics dict from the last ``agent.learn`` call, or ``{}`` when
                the buffer is not yet ready.
        """
        learn_metrics = {}
        total_samples = self.schedule.updates_per_learn * self.schedule.batch_size
        if not self.buffer.is_ready(samples = total_samples):
            return learn_metrics
        
        samples = self.buffer.sample(samples = total_samples)
        if self.schedule.updates_per_learn == 1:
            learn_metrics = self.agent.learn(
                self._step,
                samples,
                learning_epochs=self.schedule.learning_epochs,
                mini_batch_size=self.schedule.mini_batch_size
            )
            self._apply_per_update(samples, learn_metrics)

        else:
            samples = self._shuffle_sample(samples)
            for update in range(self.schedule.updates_per_learn):
                lo_idx = update * self.schedule.batch_size
                hi_idx = lo_idx + self.schedule.batch_size
                sample = {
                    k: (tree_index(v, slice(lo_idx, hi_idx)) if v is not None else None)
                    for k, v in samples.items()
                }
                learn_metrics = self.agent.learn(
                    self._step,
                    sample,
                    learning_epochs=self.schedule.learning_epochs,
                    mini_batch_size=self.schedule.mini_batch_size
                    )
                self._apply_per_update(sample, learn_metrics)
        return learn_metrics

    def train(self):
        """Run the training loop until ``schedule.is_done``.

        Steps the env, learns on the schedule's cadence, soft-updates target
        networks when present, drives callbacks, and optionally renders
        episodes. Closes the env when finished.
        """
        self._initialize_run(context="train")
        # Initialize Rich Console
        console = Console()
        start_time = time.time()

        with Live(console=console, refresh_per_second=8, transient=True) as live:
            while not self.schedule.is_done(step=self._step, episodes=self._completed_episodes.sum().item()):
                
                step_result = self.step(training=True)
                self._step += self.env.num_envs
                self.add_to_normalizers(self._prev_obs)
                self.update_schedulers()

                if self.schedule.should_learn(
                    step=self._step,
                    episodes=self._completed_episodes.sum().item(),
                    last_learn_at=self._last_learn
                ):
                    learn_metrics = self.learn()

                    # Update normalizers AFTER learning: agent.learn()
                    # re-normalizes the stored raw rollout, so the stats it
                    # uses must be the ones the policy acted under. Updating
                    # here means the next rollout is collected — and then
                    # learned — under the refreshed statistics (SB3/RSL-RL
                    # collection-time normalization semantics).
                    self.update_normalizers()
                    step_result['step_log'].update(learn_metrics)
                    # Merge n-step boundary diagnostics
                    nstep_diag = {}
                    if hasattr(self.agent, "get_nstep_diagnostics"):
                        nstep_diag = self.agent.get_nstep_diagnostics() or {}
                    nstep_wrapper = self.env._find_nstep_wrapper()
                    if nstep_wrapper and hasattr(nstep_wrapper, "get_nstep_diagnostics"):
                        wrapper_stats = nstep_wrapper.get_nstep_diagnostics() or {}
                        nstep_diag.update(wrapper_stats)
                    if nstep_diag:
                        step_result['step_log'].update(nstep_diag)  

                    # Update target networks
                    if isinstance(self.agent, HasTargetNetworks):
                        self.agent.soft_update_targets()

                    # Set learn gate
                    self._last_learn = {
                        "timestep": self._step,
                        "episode": self._completed_episodes.sum().item(),
                    }[self.schedule.learn_every_unit]

                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_train_step_end(self._step, step_result['step_log'])

                # Calculate metrics for dashboard
                elapsed = time.time() - start_time
                completed_episodes = self._completed_episodes.sum().item()
                episodes_per_sec = completed_episodes / elapsed if elapsed > 0 else 0.0
                avg_reward = sum(self._score_history) / len(self._score_history) if self._score_history else 0.0

                # Build table and update
                table = Table(
                    title="Live Training Dashboard",
                    expand=True,
                    highlight=True,
                    border_style="bold blue"
                )
                table.add_column("Steps", justify="right", style="cyan")
                table.add_column("Episodes", justify="right", style="green")
                table.add_column("Avg Reward", justify="right", style="magenta")
                table.add_column("Episodes/sec", justify="right", style="yellow")
                table.add_column("Elapsed", justify="right", style="dim")
                table.add_row(
                    f"{self._step:,}",
                    f"{completed_episodes:,}",
                    f"{avg_reward:.2f}",
                    f"{episodes_per_sec:.2f}",
                    str(timedelta(seconds=int(elapsed)))
                )
                live.update(table)

                for episode_log in step_result['episode_logs']:
                    # Reset episode score and step count to 0
                    self._episode_scores[episode_log['env']] = 0
                    self._episode_steps[episode_log['env']] = 0
                    if self.renderer and self.renderer.should_render(episode_log['episode']):
                        # Set normalizers to eval mode
                        self.set_normalizers(context="test")
                        self.renderer.render_episode(self, episode_log['episode'], self._step, context='train', seed=T.randint(high=1000000, size=(1,)).item(), render_mode='rgb_array')
                        # Set normalizers to train mode
                        self.set_normalizers(context="train")
                    if self.callbacks:
                        for callback in self.callbacks:
                            callback.on_train_epoch_end(self._step, episode_log)

            if self.callbacks:
                for episode_log in step_result['episode_logs']:
                    for callback in self.callbacks:
                        callback.on_train_end(episode_log)

        self.env.close()

    def test(self, unit: Literal["timestep", "episode"] = "episode", units: int = 1):
        """Run an evaluation loop for a fixed number of timesteps or episodes.

        Overwrites ``schedule.stop_unit`` / ``stop_units`` for this call, then
        steps without learning until that budget is exhausted.

        Args:
            unit: Whether ``units`` counts timesteps or episodes.
            units: Evaluation length in ``unit`` units.
        """
        # Update schedule for testing
        self.schedule.stop_unit = unit
        self.schedule.stop_units = units
        self._initialize_run(context="test")
        # Initialize Rich Console
        console = Console()
        start_time = time.time()

        with Live(console=console, refresh_per_second=8, transient=True) as live:
            while not self.schedule.is_done(step=self._step, episodes=self._completed_episodes.sum().item()):

                step_result = self.step(training=False)
                self._step += self.env.num_envs
                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_test_step_end(self._step, step_result['step_log'])

                # Calculate metrics for dashboard
                elapsed = time.time() - start_time
                completed_episodes = self._completed_episodes.sum().item()
                episodes_per_sec = completed_episodes / elapsed if elapsed > 0 else 0.0
                avg_reward = sum(self._score_history) / len(self._score_history) if self._score_history else 0.0

                # Build table and update
                table = Table(
                    title="Live Testing Dashboard",
                    expand=True,
                    highlight=True,
                    border_style="bold blue"
                )
                table.add_column("Steps", justify="right", style="cyan")
                table.add_column("Episodes", justify="right", style="green")
                table.add_column("Avg Reward", justify="right", style="magenta")
                table.add_column("Episodes/sec", justify="right", style="yellow")
                table.add_column("Elapsed", justify="right", style="dim")
                table.add_row(
                    f"{self._step:,}",
                    f"{completed_episodes:,}",
                    f"{avg_reward:.2f}",
                    f"{episodes_per_sec:.2f}",
                    str(timedelta(seconds=int(elapsed)))
                )
                live.update(table)

                for episode_log in step_result['episode_logs']:
                    # Reset episode score and step count to 0
                    self._episode_scores[episode_log['env']] = 0
                    self._episode_steps[episode_log['env']] = 0
                    if self.renderer and self.renderer.should_render(episode_log['episode']):
                        self.renderer.render_episode(self, episode_log['episode'], self._step, context='test', seed=T.randint(high=1000000, size=(1,)).item(), render_mode='rgb_array')
                    if self.callbacks:
                        for callback in self.callbacks:
                            callback.on_test_epoch_end(self._step, episode_log)

            if self.callbacks:
                for episode_log in step_result['episode_logs']:
                    for callback in self.callbacks:
                        callback.on_test_end(episode_log)

        self.env.close()


    def get_config(self) -> dict:
        """The entire run tree as one JSON-safe dict (the single source of truth).

        The env is serialized exactly once here (as its ``{"type", "config"}``
        spec); no sub-component re-embeds it.

        Returns:
            Mapping with ``agent``, ``env``, ``schedule``, and optional
                ``success_criterion``, ``buffer``, ``renderer``, ``callbacks``,
                plus ``log_level`` and ``save_dir``.
        """
        return {
            'agent': self.agent.get_config(),
            'env': self.env.config,
            'schedule': self.schedule.get_config(),
            'success_criterion': self.success_criterion.get_config() if self.success_criterion is not None else None,
            'buffer': self.buffer.get_config() if self.buffer is not None else None,
            'renderer': self.renderer.get_config() if self.renderer else None,
            'callbacks': [callback.get_config() for callback in self.callbacks] if self.callbacks else None,
            'log_level': self.log_level,
            'save_dir': self.save_dir,
        }

    def _trainer_state(self) -> dict:
        """Return training counters needed to resume a run.

        Returns:
            Mapping of internal counter / history fields for ``trainer_state.pt``.
        """
        return {
            "_step": self._step,
            "_best_reward": self._best_reward,
            "_last_learn": self._last_learn,
            "_completed_episodes": self._completed_episodes,
            "_episode_scores": self._episode_scores,
            "_episode_steps": self._episode_steps,
            "_score_history": list(self._score_history) if self._score_history is not None else None,
            "_success_tracker": list(self._success_tracker) if self._success_tracker is not None else None,
        }

    @staticmethod
    def _rng_state() -> dict:
        """Snapshot torch / numpy / python RNG (plus CUDA if available).

        Returns:
            Mapping with ``torch``, ``numpy``, ``python``, and optional ``cuda``.
        """
        state = {
            "torch": T.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        if T.cuda.is_available():
            state["cuda"] = T.cuda.get_rng_state_all()
        return state

    def save(self, run_dir: str | Path | None = None, *, save_buffer: bool = False) -> None:
        """Persist the full run under ``run_dir`` (defaults to ``self.save_dir``).

        Writes one JSON config (the architecture of the whole tree) plus ``.pt``
        state files mirroring the object tree:

            <run_dir>/config.json        # entire run tree, env serialized once
            <run_dir>/agent/             # weights + optimizers + normalizers + IM
            <run_dir>/trainer_state.pt   # step / episode / best-reward counters
            <run_dir>/rng.pt             # torch / numpy / python RNG
            <run_dir>/buffer.pt          # optional replay tensors (save_buffer=True)

        Args:
            run_dir: Destination directory; defaults to ``self.save_dir``.
            save_buffer: When ``True``, also write ``buffer.pt``.
        """
        run_dir = Path(run_dir) if run_dir is not None else Path(self.save_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.get_config(), f, indent=2)

        self.agent.save_state(run_dir / "agent")

        # Counters/RNG only exist once a run has been initialized.
        if self._step is not None:
            T.save(self._trainer_state(), run_dir / "trainer_state.pt")
            T.save(self._rng_state(), run_dir / "rng.pt")

        if save_buffer and self.buffer is not None:
            self.buffer.save_state(run_dir / "buffer.pt")

    @classmethod
    def load(
        cls,
        run_dir: str | Path,
        *,
        env: EnvWrapper | None = None,
        load_weights: bool = True,
        load_buffer: bool = False,
        log_level: str | None = None,
    ) -> "Trainer":
        """Rebuild a fully wired Trainer from a directory written by ``save``.

        See [Trainer.save][phoenx.trainer.Trainer.save] for the on-disk layout.

        Args:
            run_dir: Directory containing ``config.json`` and the state files.
            env: Optional live env to reuse instead of rebuilding from config.
            load_weights: Restore model weights (``False`` = architecture only).
            load_buffer: Also restore the replay buffer from ``buffer.pt``.
            log_level: Logger level for the rebuilt Trainer.

        Returns:
            A Trainer ready to ``.train()`` (resumes counters/RNG) or ``.test()``.
        """
        run_dir = Path(run_dir)
        with open(run_dir / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        if env is None:
            env = EnvWrapper.from_json(json.dumps(config["env"]))

        agent = build_agent(config["agent"], env)
        agent.load_state(run_dir / "agent", load_weights=load_weights)

        schedule = TrainingSchedule(**config["schedule"])
        success_criterion = SuccessCriterion(**config["success_criterion"]) if config.get("success_criterion") else None
        if load_buffer:
            buffer = cls._build_buffer(config.get("buffer"), env)
        else:
            buffer = None
        renderer = Renderer(**config["renderer"]) if config.get("renderer") else None
        callbacks = (
            [build_callback(cb) for cb in config["callbacks"]]
            if config.get("callbacks") else None
        )

        log_level = log_level if log_level is not None else config.get('log_level', 'INFO')

        trainer = cls(
            agent=agent,
            env=env,
            schedule=schedule,
            success_criterion=success_criterion,
            buffer=buffer,
            renderer=renderer,
            callbacks=callbacks,
            log_level=log_level,
            save_dir=str(run_dir),
        )

        # Stage counters/RNG; applied by _initialize_run once the env is reset.
        state_path = run_dir / "trainer_state.pt"
        if state_path.exists():
            trainer._resume_state = T.load(state_path, map_location=agent.device, weights_only=False)
        rng_path = run_dir / "rng.pt"
        if rng_path.exists():
            trainer._resume_rng = T.load(rng_path, map_location="cpu", weights_only=False)

        
        buffer_restored = load_buffer and buffer is not None and (run_dir / "buffer.pt").exists()
        if buffer_restored:
            buffer.load_state(run_dir / "buffer.pt")
        trainer._resume_buffer_restored = buffer_restored

        return trainer

    @staticmethod
    def _build_buffer(buffer_config: dict | None, env: EnvWrapper) -> Buffer | None:
        """Reconstruct a buffer from its saved ``{"type", "config"}`` block.

        Args:
            buffer_config: Serialized buffer spec, or ``None`` / empty to skip.
            env: Live environment injected into the buffer (and any hindsight
                relabeler) instead of rebuilding from config.

        Returns:
            Rebuilt ``Buffer`` instance, or ``None`` when ``buffer_config`` is
                empty.
        """
        if not buffer_config:
            return None
        kwargs = dict(buffer_config.get("config", {}))
        kwargs.pop("env", None)  # env is injected live, never rebuilt from config
        kwargs["env"] = env
        hindsight = kwargs.get("hindsight")
        if hindsight is not None:
            hindsight_kwargs = dict(hindsight.get("config", hindsight))
            hindsight_kwargs.pop("env", None)
            hindsight_kwargs["env"] = env
            kwargs["hindsight"] = HindsightRelabeler(**hindsight_kwargs)
        return Buffer.create_instance(buffer_config["type"], **kwargs)
