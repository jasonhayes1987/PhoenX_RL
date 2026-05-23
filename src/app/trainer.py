from rich.live import Live
from rich.table import Table
from rich.console import Console
import time
from datetime import timedelta
from dataclasses import dataclass, replace
import logging
from typing import Literal, Any
import numpy as np
import torch as T
from torch.distributions import Distribution, Categorical
from collections import deque
from abc import abstractmethod

from .torch_utils import set_seed
from .rl_callbacks import Callback, WandbCallback
from .rl_agents import Agent, HasTargetNetworks
from .env_wrapper import EnvWrapper, Observation, VectorNStepReward
from .buffer import Buffer, ReplayBuffer, PrioritizedReplayBuffer, RolloutBuffer, TrajectoryBuffer
from .renderer import Renderer
from .logging_config import get_logger, configure_logging

@dataclass
class TrainingSchedule:
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
        if self.stop_unit == "timestep":
            return step >= self.stop_units
        return episodes >= self.stop_units
    
    def should_learn(
        self, *,
        step: int,
        episodes: int,
        last_learn_at: int,
    ) -> bool:
        if step < self.warmup_steps:
            return False
        progress = {
            "timestep": step,
            "episode": episodes,
        }[self.learn_every_unit]
        return progress >= last_learn_at + self.learn_every

    def get_config(self) -> dict:
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

class Trainer:
    def __init__(
        self,
        agent:Agent,
        env:EnvWrapper,
        buffer:Buffer,
        schedule:TrainingSchedule,
        renderer:Renderer|None = None,
        callbacks:list[Callback]|None = None,
        log_level: str = 'INFO',
        save_dir: str = 'models/',
    ):
        
        self.agent = agent
        self.env = env
        self.buffer = buffer
        self.schedule = schedule
        self.renderer = renderer
        self.callbacks = callbacks
        self.save_dir = save_dir

        # Set Agent and Renderer save dir
        self.agent.save_dir = self.save_dir
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

    def _initialize_callbacks(self):
        """
        Initialize and configure callbacks for logging and monitoring.

        """
        try:
            if self.callbacks:
                for callback in self.callbacks:
                    callback._config(self.get_config())
                    if isinstance(callback, WandbCallback):
                        self._wandb = True

        except Exception as e:
            raise ValueError(f"Error initializing callbacks: {e}")

    def _initialize_run(self, context: Literal["train", "test"], **kwargs: Any):
        """
        Initializes the environment, seeds, and tracking variables for training.
        Args:
            context (Literal["train", "test"]): Context of the run.
            **kwargs: Additional keyword arguments for the run.
        """
        if self._initialized:
            return
        
        # Set models to train mode if training, else evaluation mode
        for name in ['policy', 'value', 'critic', 'critic_a', 'critic_b']:
            model = getattr(self.agent, name, None)
            if model:
                if context == "train":
                    model.train()
                    model.logger.debug(f"Set {name} to train mode")
                elif context == "test":
                    model.eval()
                    model.logger.debug(f"Set {name} to eval mode")


        # Set target models to eval mode
        for name in ['target_policy', 'target_critic', 'target_critic_a', 'target_critic_b']:
            model = getattr(self.agent, name, None)
            if model:
                model.eval()
                model.logger.debug(f"Set {name} to eval mode")

        # Set VectorNStepReward wrapper Intrinsic Motivation pointer
        im = getattr(self.agent, 'intrinsic_motivation', None)
        if im is not None:
            nstep_wrapper = self._find_nstep_wrapper(self.env)
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

            models = [model for model in [getattr(self.agent, "policy", None),
                                          getattr(self.agent, "critic", None),
                                          getattr(self.agent, "critic_a", None),
                                          getattr(self.agent, "critic_b", None),
                                          getattr(self.agent, "value", None),
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

    def _find_nstep_wrapper(self, env: EnvWrapper) -> VectorNStepReward | None:
        while env is not None:
            if isinstance(env, VectorNStepReward):
                return env
            env = getattr(env, 'env', None)

    def _iter_normalizers(self):
        """Yields (name, normalizer) for every normalizer the agent actually has."""
        for name in ("state_normalizer", "goal_normalizer",
                    "reward_normalizer", "advantage_normalizer"):
            norm = getattr(self.agent, name, None)
            if norm is not None:
                yield name, norm

    def _apply_per_update(self, sample: dict, learn_metrics: dict) -> None:
        """If using a PrioritizedReplayBuffer, push TD errors back as priorities
        and (optionally) collect PER diagnostics for logging."""
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
        # pb = self.buffer  # PrioritizedReplayBuffer
        actual_size = min(self.buffer.counter, self.buffer.buffer_size)
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
        """Sets the normalizers to train or eval mode."""
        if context not in ("train", "test"):
            raise ValueError(f"Invalid context: {context}")
        for _, norm in self._iter_normalizers():
            norm.train() if context == "train" else norm.eval()
        
        # # Set Intrinsic Motivation normalizers if present
        # im = getattr(self.agent, 'intrinsic_motivation', None)
        # if im is not None:
        #     im.set_normalizers_mode(context)

    def add_to_normalizers(self, obs: Observation):
        """
        Add relavent data from obs to the normalizers.

        Args:
            obs: Observation to feed.
        """
        for name, norm in self._iter_normalizers():
            if name == "state_normalizer":
                norm.add(obs.states.to(device=norm.device))

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
        """Updates the normalizers."""
        for _, norm in self._iter_normalizers():
            norm.update()

        # # Update Intrinsic Motivation normalizers if present
        # im = getattr(self.agent, 'intrinsic_motivation', None)
        # if im is not None:
        #     im.update_normalizers()

    def normalize_observation(
        self, obs: Observation)->Observation:
        """Normalizes the observation for the agent.

        Args:
            obs (Observation): Observation to normalize.
        
        Returns:
            Observation: Normalized observation.
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
        schedulers = [
            getattr(self.agent, name, None)
            for name in ("entropy_schedule", "noise_schedule", "target_noise_schedule")
        ] + [
            getattr(getattr(self.agent, "policy", None), name, None)
            for name in ("temperature_schedule", "lr_scheduler")
        ] + [
            getattr(getattr(self.agent, model, None), "lr_scheduler", None)
            for model in ("value", "critic", "critic_b")
        ] + [
            getattr(getattr(self.agent, "intrinsic_motivation", None), "reward_scheduler", None)
        ]
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
        """
        Performs a single step of training/testing.
        
        Args:
        training: bool: Whether the step is for training or testing.
        
        Returns:
        dict: A dictionary containing the step metrics.
        """

        step_log = {}
        episode_logs = []

        # Normalize observations and goals if normalizers
        obs_norm = self.normalize_observation(self._prev_obs)
        actions = self.get_action(obs_norm.states, obs_norm.goals, context='train' if training else 'test')
        # Take action in environment and get new Observation
        observation = self.env.step(actions)
        # If Agent uses Intrinsic Motivation, calculate intrinsic rewards to store in buffer
        im = getattr(self.agent, 'intrinsic_motivation', None)
        if im is not None:
            # Pass normalized states to IM if present
            next_obs_norm = self.normalize_observation(observation)
            intrinsic_rewards = im.compute_rollout_reward(obs_norm.states, next_obs_norm.states, actions, env_indices = T.arange(self.env.num_envs, device=im.device))
            # observation = replace(observation, intrinsic_rewards=intrinsic_rewards)
        else:
            intrinsic_rewards = T.zeros_like(observation.rewards)
        observation = replace(observation, intrinsic_rewards=intrinsic_rewards)

        dones = T.logical_or(observation.terminations, observation.truncations)
        valid_steps = ~self._prev_done
       
        
        # Add transitions to the buffer (non-normalized)
        self.buffer.record(observation, prev_observation = self._prev_obs, actions = actions, prev_dones = self._prev_done)

        # Increment episode steps and rewards
        self._episode_steps[valid_steps] += 1
        self._episode_scores[valid_steps] += observation.rewards[valid_steps].flatten()

        # Add step metrics to step log
        step_log.update({
            'step_reward': observation.rewards[valid_steps].mean().item(),
            'step_intrinsic_reward': observation.intrinsic_rewards[valid_steps].mean().item() if self.agent.intrinsic_motivation else 0.0
        })
        
        # Check if any env is done
        done_episodes = T.logical_or(observation.terminations, observation.truncations).nonzero(as_tuple=False).flatten()

        for i in done_episodes:
            self._completed_episodes[i] += 1
            self._score_history.append(float(self._episode_scores[i].item()))
            avg_reward = sum(self._score_history) / len(self._score_history)
            # check if best reward
            episode_log = {
                'env': i,
                'episode': int(self._completed_episodes.sum()),
                'episode_steps': int(self._episode_steps[i].item()),
                'episode_reward': round(float(self._episode_scores[i]), 2),
                'avg_reward': round(float(avg_reward), 2)
            }
            if training:
                # Check if best reward
                if avg_reward > self._best_reward:
                    self._best_reward = avg_reward
                    self.agent.save()
            episode_logs.append(episode_log)

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
        """
        Select an action based on the current policy.

        Args:
            states: np.ndarray | T.Tensor: The current states.
            goals: np.ndarray | T.Tensor | None: The current goals.
            context: str: The context of the action (train, test).
        
        Returns:
            T.Tensor: actions.
        """

        return self.agent.act(
            states,
            goals,
            context,
            step = self._step,
            warmup = self.schedule.warmup_steps
        )

    def learn(self)->dict:
        """
        Calls Agent.learn() schedule.update times, passing samples from the buffer.
        
        Returns:
            dict: A dictionary containing the learn metrics.
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
            for update in range(self.schedule.updates_per_learn):
                lo_idx = update * self.schedule.batch_size
                hi_idx = lo_idx + self.schedule.batch_size
                sample = {k: (v[lo_idx:hi_idx] if v is not None else None) for k, v in samples.items()}
                learn_metrics = self.agent.learn(
                    self._step,
                    sample,
                    learning_epochs=self.schedule.learning_epochs,
                    mini_batch_size=self.schedule.mini_batch_size
                    )
                self._apply_per_update(sample, learn_metrics)
        return learn_metrics

    def train(self):
        """Trains Agent following the Schedule."""
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
                    # Update normalizers
                    self.update_normalizers()

                    learn_metrics = self.learn()
                    step_result['step_log'].update(learn_metrics)

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

    def get_config(self) -> dict:
        return {
            'agent': self.agent.get_config(),
            'env': self.env.config,
            'buffer': self.buffer.get_config(),
            'schedule': self.schedule.get_config(),
            'renderer': self.renderer.get_config() if self.renderer else None,
            'callbacks': [callback.get_config() for callback in self.callbacks] if self.callbacks else None,
            'save_dir': self.save_dir,
        }