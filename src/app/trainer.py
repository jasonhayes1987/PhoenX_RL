from rich.live import Live
from rich.table import Table
from rich.console import Console
import time
from datetime import timedelta
from dataclasses import dataclass
import logging
from typing import Literal, Any
import numpy as np
import torch as T
from torch.distributions import Distribution, Categorical
from collections import deque
from abc import abstractmethod

from .torch_utils import set_seed
from .rl_callbacks import Callback, WandbCallback
from .rl_agents import Agent
from .env_wrapper import EnvWrapper
from .buffer import Buffer, ReplayBuffer, PrioritizedReplayBuffer, RolloutBuffer, TrajectoryBuffer
from .renderer import Renderer
from .logging_config import get_logger, configure_logging

@dataclass
class TrainingSchedule:
    # Training Length
    stop_unit:   Literal["timestep", "episode"] = "timestep"
    stop_units:  int = 1_000_000
    # Learn Frequency
    learn_every_unit: Literal["timestep", "episode"] = "timestep"
    learn_every:      int = 1
    # Per learn() call
    updates_per_learn: int = 1
    batch_size:        int = 1
    learning_epochs:   int = 1
    # Optional warmup gate (no learning until this many steps)
    warmup_steps: int = 0
    seed: int | None = None

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

# @dataclass
# class Schedule:
#     seed: int | None = None

# @dataclass
# class OnPolicySchedule(Schedule):
#     # Training length
#     unit: Literal["timestep", "episode"] = 'episode'
#     units: int = 1000

#     # learning Params
#     learn_unit: Literal["timestep", "trajectory"] = 'trajectory'
#     learn_units: int = 10

#     # Algo specific (ie. PPO)
#     batch_size: int|None = None
#     learning_epochs: int|None = None

#     # Set seed
#     seed: int|None = None

# @dataclass
# class OffPolicySchedule(Schedule):
#     # Training length
#     unit: Literal["timestep", "episode"] = 'episode'
#     units: int = 1000

#     # learning Params
#     # cycles: int = 10
#     episodes: int = 16
#     updates: int = 40
#     batch_size: int = 128

#     # Set seed
#     seed: int|None = None

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
    ):
        
        self.agent = agent
        self.env = env
        self.buffer = buffer
        self.schedule = schedule
        self.renderer = renderer
        self.callbacks = callbacks

        # Initialize internal attributes
        self.logger = get_logger(self.__class__.__name__, level=log_level.upper())
        self._initialized = False
        self._wandb = False
        self._step = None
        self._prev_obs = None
        self._best_reward = None
        self._completed_episodes = None
        self._episode_scores = None
        self._score_history = None
        self._last_learn = None

    def initialize_callbacks(self):
        """
        Initialize and configure callbacks for logging and monitoring.

        """
        try:
            if self.callbacks:
                for callback in self.callbacks:
                    callback._config(self.agent)
                    if isinstance(callback, WandbCallback):
                        self._wandb = True

        except Exception as e:
            raise ValueError(f"Error initializing callbacks: {e}")

    def initialize_run(self, context: Literal["train", "test"], **kwargs: Any):
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

        # Set normalizers to train or eval mode
        if context == "train":
            if self.agent.state_normalizer:
                self.agent.state_normalizer.train()
            if self.agent.goal_normalizer:
                self.agent.goal_normalizer.train()
            if hasattr(self.agent, 'advantage_normalizer') and self.agent.advantage_normalizer:
                self.agent.advantage_normalizer.train()
        
        elif context == "test":
            if self.agent.state_normalizer:
                self.agent.state_normalizer.eval()
            if self.agent.goal_normalizer:
                self.agent.goal_normalizer.eval()
            if hasattr(self.agent, 'advantage_normalizer') and self.agent.advantage_normalizer:
                self.agent.advantage_normalizer.eval()

        # Set internal attributes
        seed = self.schedule.seed if self.schedule.seed else T.randint(2**31-1, (1,)).item()
        set_seed(seed)
        observation = self.env.reset(seed=seed)

        # Warmup normalizers if exist
        if self.agent.state_normalizer or self.agent.goal_normalizer:
            if context == "train":
                for _ in range(self.agent.state_normalizer.warmup_steps):
                    observation = self.env.sample_observation()
                    if self.agent.state_normalizer:
                        # self.agent.state_normalizer.train()
                        self.agent.state_normalizer.add(observation.states.to(device=self.agent.state_normalizer.device))
                        self.agent.state_normalizer.update()
                    if self.agent.goal_normalizer:
                        # self.agent.goal_normalizer.train()
                        # Concatenate current goals and achieved goals into 1 tensor
                        goals = T.cat([observation.goals, observation.ach_goals], dim=0).to(device=self.agent.goal_normalizer.device)
                        self.agent.goal_normalizer.add(goals)
                        self.agent.goal_normalizer.update()
                    # if hasattr(self.agent, 'advantage_normalizer') and self.agent.advantage_normalizer:
                    #     if context == "train":
                    #         self.agent.advantage_normalizer.train()
                    # Reset environments
                observation = self.env.reset()
            # else:
            #     if self.agent.state_normalizer:
            #         self.agent.state_normalizer.eval()
            #     if self.agent.goal_normalizer:
            #         self.agent.goal_normalizer.eval()
                # if hasattr(self.agent, 'advantage_normalizer') and self.agent.advantage_normalizer:
                #     self.agent.advantage_normalizer.eval()

        # Set callbacks
        if self.callbacks:
            self.initialize_callbacks()
            config = self.agent.get_config()
            config.update({'num_envs': self.env.num_envs, 'seed': seed})
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

    def set_normalizers(self, context: Literal["train", "test"]):
        """Sets the normalizers to train or eval mode."""
        if context == "train":
            if self.agent.state_normalizer:
                self.agent.state_normalizer.train()
            if self.agent.goal_normalizer:
                self.agent.goal_normalizer.train()
            if self.agent.reward_normalizer:
                self.agent.reward_normalizer.train()
        elif context == "test":
            if self.agent.state_normalizer:
                self.agent.state_normalizer.eval()
            if self.agent.goal_normalizer:
                self.agent.goal_normalizer.eval()
            if self.agent.reward_normalizer:
                self.agent.reward_normalizer.eval()
        else:
            raise ValueError(f"Invalid context: {context}")

    def add_to_normalizers(self):
        """Adds to the normalizers."""
        if self.agent.state_normalizer:
            self.agent.state_normalizer.add(self._prev_obs.states.to(device=self.agent.state_normalizer.device))
        if self.agent.goal_normalizer:
            self.agent.goal_normalizer.add(T.cat([self._prev_obs.goals, self._prev_obs.ach_goals], dim=0).to(device=self.agent.goal_normalizer.device))
        if self.agent.reward_normalizer:
            self.agent.reward_normalizer.add(self._prev_obs.rewards, T.logical_or(self._prev_obs.terminations, self._prev_obs.truncations))

    def update_normalizers(self):
        """Updates the normalizers."""
        if self.agent.state_normalizer:
            self.agent.state_normalizer.update()
        if self.agent.goal_normalizer:
            self.agent.goal_normalizer.update()
        if self.agent.reward_normalizer:
            self.agent.reward_normalizer.update()
        if hasattr(self.agent, 'advantage_normalizer') and self.agent.advantage_normalizer:
            self.agent.advantage_normalizer.update()

    def normalize_inputs(
        self,
        states: np.ndarray | T.Tensor,
        goals: np.ndarray | T.Tensor | None = None,
        ach_goals: np.ndarray | T.Tensor | None = None
    )->tuple[T.Tensor, T.Tensor | None, T.Tensor | None]:
        """Normalizes the states and goals for the agent.

        Args:
            states (np.ndarray | T.Tensor): States to normalize.
            goals (np.ndarray | T.Tensor | None): Goals to normalize.
            ach_goals (np.ndarray | T.Tensor | None): Achieved goals to normalize.
        
        Returns:
            tuple: Tuple of normalized states, goals, and achieved goals as Tensors.
        """
        if self.agent.state_normalizer:
            states = self.agent.state_normalizer.normalize(states)
        if hasattr(self.agent, 'goal_normalizer') and self.agent.goal_normalizer:
            if goals is not None:
                goals = self.agent.goal_normalizer.normalize(goals)
            if ach_goals is not None:
                ach_goals = self.agent.goal_normalizer.normalize(ach_goals)
        
        return states, goals, ach_goals

    def update_schedulers(self):
        """Updates the schedulers."""
        if hasattr(self.agent, 'entropy_schedule') and self.agent.entropy_schedule:
            self.agent.entropy_schedule.step(self.env.num_envs)
        if hasattr(self.agent.policy, 'temperature_schedule') and self.agent.policy.temperature_schedule:
            self.agent.policy.temperature_schedule.step(self.env.num_envs)
        if hasattr(self.agent.policy, 'lr_scheduler') and self.agent.policy.lr_scheduler:
            self.agent.policy.lr_scheduler.step(self.env.num_envs) 
        if hasattr(self.agent, 'value') and self.agent.value and hasattr(self.agent.value, 'lr_scheduler') and self.agent.value.lr_scheduler:
            self.agent.value.lr_scheduler.step(self.env.num_envs)
        if hasattr(self.agent, 'critic') and self.agent.critic and hasattr(self.agent.critic, 'lr_scheduler') and self.agent.critic.lr_scheduler:
            self.agent.critic.lr_scheduler.step(self.env.num_envs)
        if hasattr(self.agent, 'critic_b') and self.agent.critic_b and hasattr(self.agent.critic_b, 'lr_scheduler') and self.agent.critic_b.lr_scheduler:
            self.agent.critic_b.lr_scheduler.step(self.env.num_envs)
        if hasattr(self.agent, 'noise_schedule') and self.agent.noise_schedule:
            self.agent.noise_schedule.step(self.env.num_envs)
        if hasattr(self.agent, 'target_noise_schedule') and self.agent.target_noise_schedule:
            self.agent.target_noise_schedule.step(self.env.num_envs)

    # def training_complete(self)->bool:
    #     """Checks if the training is complete."""
    #     if self.schedule.unit == 'timestep':
    #         return self._step >= self.schedule.units
    #     elif self.schedule.unit == 'episode':
    #         return self._completed_episodes.sum() >= self.schedule.units
    #     else:
    #         raise ValueError(f"Invalid unit: {self.schedule.unit}")

    def step(self, training: bool = True):
        """
        Performs a single step of training/testing.
        
        Args:
        training: bool: Whether the step is for training or testing.
        
        Returns:
        dict: A dictionary containing the step metrics.
        """
        # print(f'###STEP {self._step}###')
        step_log = {}
        episode_logs = []

        # Normalize observations and goals if normalizers
        obs_norm, goals_norm, ach_goals_norm = self.normalize_inputs(self._prev_obs.states, self._prev_obs.goals, self._prev_obs.ach_goals)
        actions = self.get_action(obs_norm, goals_norm, context='train' if training else 'test')
        observation = self.env.step(actions)

        dones = T.logical_or(observation.terminations, observation.truncations)
        valid_steps = ~self._prev_done
       
        
        # Add normalized transitions to the buffer
        self.buffer.record(observation, prev_observation = self._prev_obs, actions = actions, prev_dones = self._prev_done)

        # Increment episode steps and rewards
        self._episode_steps[valid_steps] += 1
        self._episode_scores[valid_steps] += observation.rewards[valid_steps].flatten()

        # Add step metrics to step log
        step_log.update({
            'step_reward': observation.rewards[valid_steps].mean().item()
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

    @abstractmethod
    def train(self):
        """Trains the Agent."""
        raise NotImplementedError("Subclasses must implement train.")

class OnPolicyTrainer(Trainer):
    def __init__(
        self,
        agent: Agent,
        env: EnvWrapper,
        buffer: RolloutBuffer|TrajectoryBuffer,
        schedule: TrainingSchedule,
        renderer: Renderer | None = None,
        callbacks: list[Callback] | None = None,
        log_level: str = 'INFO',
    ):
        super().__init__(agent, env, buffer, schedule, renderer, callbacks, log_level)
        self._learn_counter = 0

    # def should_learn(self)->bool:
    #     """Checks if the agent should learn based on the schedule."""
    #     if self.schedule.learn_unit not in ['timestep', 'trajectory']:
    #         raise ValueError(f"Invalid learn unit: {self.schedule.learn_unit}")
    #     elif self.schedule.learn_unit == 'timestep':
    #         self._learn_counter += self.env.num_envs
    #         if self._learn_counter >= self.schedule.learn_units:
    #             self._learn_counter = 0
    #             return True
    #     else: # self.schedule.learn_unit == 'trajectory'
    #         if self._completed_episodes.sum() >= self.schedule.learn_units * (self._learn_counter + 1):
    #             self._learn_counter += 1
    #             return True
    #     return False

    def learn(self)->dict:
        """Calls Agent.learn() passing samples from the buffer.
        Will also pass batch_size and learning_epochs from the schedule if they are set.
        
        Returns:
            dict: A dictionary containing the learn metrics.
        """
        learn_metrics = {}
        if self.buffer.is_ready():
            learn_metrics = self.agent.learn(
                self._step, self.buffer.sample(),
                batch_size=self.schedule.batch_size,
                learning_epochs=self.schedule.learning_epochs
            )
        return learn_metrics

    # def get_action(
    #     self,
    #     states: np.ndarray | T.Tensor,
    #     goals: np.ndarray | T.Tensor | None = None,
    #     context: str = 'train'
    # ) -> T.Tensor:
    #     """
    #     Select an action based on the current policy.
    #     Returns actions that are already scaled to the environment's action space.

    #     Args:
    #         states: np.ndarray | T.Tensor: The current states.
    #         goals: np.ndarray | T.Tensor | None: The current goals.
    #         context: str: The context of the action (train, test).
        
    #     Returns:
    #         T.Tensor: actions.
    #     """

    #     return self.agent.act(states, goals, context=context)

        # if context == 'test':
        #     with T.no_grad():
        #         dist = self.agent.policy(states, goals)
        # else:
        #     dist = self.agent.policy(states, goals)
        
        # if context == 'train':
        #     actions = dist.sample()
        # elif context == 'test':
        #     actions = self.agent.policy.get_mean_actions(dist)
        # else:
        #     raise ValueError(f"Invalid context: {context}")

        # return actions

    # def step(self, training: bool = True):
    #     """
    #     Performs a single step of training/testing.
        
    #     Args:
    #     training: bool: Whether the step is for training or testing.
        
    #     Returns:
    #     dict: A dictionary containing the step metrics.
    #     """
    #     # print(f'###STEP {self._step}###')
    #     step_log = {}
    #     episode_logs = []

    #     # Normalize observations and goals if normalizers
    #     obs_norm, goals_norm, ach_goals_norm = self.normalize_inputs(self._prev_obs.states, self._prev_obs.goals, self._prev_obs.ach_goals)
    #     actions = self.get_action(obs_norm, goals_norm, context='train' if training else 'test')
    #     observation = self.env.step(actions)

    #     dones = T.logical_or(observation.terminations, observation.truncations)
    #     valid_steps = ~self._prev_done
       
        
    #     # Add normalized transitions to the buffer
    #     self.buffer.record(observation, self._prev_obs, actions, self._prev_done)
    #     # self.buffer.add(
    #     #     states=self._cur_obs.states,
    #     #     actions=actions,
    #     #     rewards=observation.rewards,
    #     #     next_states=observation.states,
    #     #     terminations=observation.terminations,
    #     #     truncations=observation.truncations,
    #     #     state_achieved_goals=self._cur_obs.ach_goals if self._cur_obs.ach_goals is not None else None,
    #     #     next_state_achieved_goals=observation.ach_goals if observation.ach_goals is not None else None,
    #     #     desired_goals=self._cur_obs.goals if self._cur_obs.goals is not None else None,
    #     #     first_steps=self._prev_done,
    #     # )

    #     # Increment episode steps and rewards
    #     self._episode_steps[valid_steps] += 1
    #     self._episode_scores[valid_steps] += observation.rewards[valid_steps].flatten()

    #     # Add step metrics to step log
    #     step_log.update({
    #         'step_reward': observation.rewards[valid_steps].mean().item()
    #     })
        
    #     # Check if any env is done
    #     done_episodes = T.logical_or(observation.terminations, observation.truncations).nonzero(as_tuple=False).flatten()

    #     for i in done_episodes:
    #         self._completed_episodes[i] += 1
    #         self._score_history.append(float(self._episode_scores[i].item()))
    #         avg_reward = sum(self._score_history) / len(self._score_history)
    #         # check if best reward
    #         episode_log = {
    #             'env': i,
    #             'episode': int(self._completed_episodes.sum()),
    #             'episode_steps': int(self._episode_steps[i].item()),
    #             'episode_reward': round(float(self._episode_scores[i]), 2),
    #             'avg_reward': round(float(avg_reward), 2)
    #         }
    #         if training:
    #             # Check if best reward
    #             if avg_reward > self._best_reward:
    #                 self._best_reward = avg_reward
    #                 self.agent.save()
    #         episode_logs.append(episode_log)

    #     # set _cur_obs to observation
    #     self._prev_done = dones.clone()
    #     self._prev_obs = observation

    #     return{
    #     'step_log': step_log,
    #     'episode_logs': episode_logs,
    # }

    def train(self):
        """Trains On-Policy Agent following the Schedule."""
        self.initialize_run(context="train")
        # Set internal learn counter for timestep based learning

        # Initialize Rich Console
        console = Console()
        start_time = time.time()

        with Live(console=console, refresh_per_second=8, transient=True) as live:
            while not self.schedule.is_done(step=self._step, episodes=self._completed_episodes.sum().item()):

                step_result =self.step(training=True)
                self._step += self.env.num_envs
                self.add_to_normalizers()
                self.update_schedulers()

                if self.schedule.should_learn(
                    step=self._step,
                    episodes=self._completed_episodes.sum().item(),
                    last_learn_at=self._last_learn
                ):
                    learn_metrics = self.learn()
                    step_result['step_log'].update(learn_metrics)

                    # Update normalizers
                    self.update_normalizers()

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

class OffPolicyTrainer(Trainer):
    def __init__(
        self,
        agent: Agent,
        env: EnvWrapper,
        buffer: ReplayBuffer|PrioritizedReplayBuffer,
        schedule: TrainingSchedule,
        renderer: Renderer | None = None,
        callbacks: list[Callback] | None = None,
        log_level: str = 'INFO',
    ):
        super().__init__(agent, env, buffer, schedule, renderer, callbacks, log_level)
        self._learn_counter = 0

    def reset_noise(self):
        """
        Resets any noise objects with a reset function on the agent
        """
        if hasattr(self.agent, 'noise'):
            if hasattr(self.agent.noise, 'reset'):
                self.agent.noise.reset()
        if hasattr(self.agent, 'target_noise'):
            if hasattr(self.agent.target_noise, 'reset'):
                self.agent.target_noise.reset()

    def learn(self)->dict:
        """
        Calls Agent.learn() schedule.update times, passing samples from the buffer.
        
        Returns:
            dict: A dictionary containing the learn metrics.
        """
        learn_metrics = {}
        total_samples = self.schedule.updates_per_learn * self.schedule.batch_size
        if self.buffer.is_ready(samples = total_samples):
            samples = self.buffer.sample(samples = total_samples)
            if self.schedule.updates_per_learn == 1:
                learn_metrics = self.agent.learn(self._step, samples)
            else:
                for update in range(self.schedule.updates_per_learn):
                    lo_idx = update * self.schedule.batch_size
                    hi_idx = lo_idx + self.schedule.batch_size
                    sample = {k: (v[lo_idx:hi_idx] if v is not None else None) for k, v in samples.items()}
                    learn_metrics = self.agent.learn(self._step, sample)
        return learn_metrics

    # def get_action(self,
    #     states: np.ndarray|T.Tensor,
    #     goals: np.ndarray|T.Tensor|None=None,
    #     context: str = 'train'
    # )->T.Tensor:
    #     """
    #     Select an action based on the current policy.

    #     Args:
    #         states: np.ndarray | T.Tensor: The current states.
    #         goals: np.ndarray | T.Tensor | None: The current goals.
    #         context: str: The context of the action (train, test).
        
    #     Returns:
    #         T.Tensor: actions.
    #     """

    #     return self.agent.act(states, goals, context=context, step=self._step)
        # # raw_actions = None
        
        # # If training
        # if context == 'train':
        #     # If warmup, sample random action from action space
        #     if (self._step is not None) and (self._step <= self.agent.warmup):
        #         return self.env.action_space.sample()
        #     # if random number is less than epsilon, sample random action
        #     if np.random.random() < self.agent.action_epsilon:
        #         return self.env.action_space.sample()
        #     # otherwise, sample action from policy
        #     else:
        #         noise = self.agent.noise(self.env.action_space.shape)
        #         # Apply noise clipping if needed
        #         if self.agent.noise_clip > 0:
        #             noise = noise.clamp(-self.agent.noise_clip, self.agent.noise_clip)
        #         # Apply noise schedule if needed
        #         if self.agent.noise_schedule:
        #             noise *= self.agent.noise_schedule.get_factor()
                
        #         with T.no_grad():
        #             _, actions = self.agent.policy(states, goals)
                
        #         # Convert the action space bounds to a tensor on the same device
        #         action_space_high = T.tensor(self.env.action_space.high, dtype=T.float32, device=self.agent.policy.device)
        #         action_space_low = T.tensor(self.env.action_space.low, dtype=T.float32, device=self.agent.policy.device)
        #         actions = (actions + noise).clip(action_space_low, action_space_high)

        #         return actions.detach()

        # else: # context == 'test'
        #     with T.no_grad():
        #         _, actions = self.agent.target_policy(states, goals)
        #     return actions.detach()

        # else: # learn
        #     raw_actions, squashed_actions = self.agent.policy(states, goals)
        #     return squashed_actions, raw_actions

    # def step(self, training: bool = True):
    #     """
    #     Performs a single step of training/testing.

    #     Args:
    #     training: bool: Whether the step is for training or testing.

    #     Returns:
    #     dict: A dictionary containing the step metrics.
    #     """
    #     step_log = {}
    #     episode_logs = []

    #     # Normalize observations and goals if normalizers
    #     obs_norm, goals_norm, ach_goals_norm = self.normalize_inputs(self._prev_obs.states, self._prev_obs.goals, self._prev_obs.ach_goals)

    #     actions = self.get_action(obs_norm, goals_norm, context='train' if training else 'test')
    #     observation = self.env.step(actions)

    #     # if observation.n_step_trajectory is not None:
    #     #     self.buffer.add(**observation.n_step_trajectory)
    #     self.buffer.record(observation, self._prev_obs, actions, self._prev_done)
    #     # else:
    #     #     raise ValueError("n-step trajectory is None. Must use VectorNStepReward wrapper when using OffPolicyTrainer.")
    #     # else:
    #     #     # Add single transitions to the buffer
    #     #     self.buffer.add(
    #     #         states=self._cur_obs.states,#[valid],
    #     #         actions=actions,#[valid],
    #     #         rewards=observation.rewards,#[valid],
    #     #         next_states=observation.states,#[valid],
    #     #         terminations=observation.terminations,#[valid],
    #     #         truncations=observation.truncations,#[valid],
    #     #         state_achieved_goals=self._cur_obs.ach_goals,#[valid],
    #     #         next_state_achieved_goals=observation.ach_goals,#[valid],
    #     #         desired_goals=self._cur_obs.goals#[valid],
    #     #     )

    #     # Increment episode step and rewards
    #     self._episode_steps += 1
    #     self._episode_scores += observation.rewards.flatten()
    #     # Add step metrics to step log
    #     step_log.update({
    #         'step_reward': observation.rewards.mean().item()
    #     })

    #     # Check if any env is done
    #     done_episodes = T.logical_or(observation.terminations, observation.truncations).nonzero(as_tuple=False).flatten()

    #     for i in done_episodes:
    #         self._completed_episodes[i] += 1
    #         self._score_history.append(float(self._episode_scores[i].item()))
    #         avg_reward = sum(self._score_history) / len(self._score_history)
    #         # check if best reward
    #         episode_log = {
    #             'env': i,
    #             'episode': int(self._completed_episodes.sum()),
    #             'episode_steps': int(self._episode_steps[i].item()),
    #             'episode_reward': round(float(self._episode_scores[i]), 2),
    #             'avg_reward': round(float(avg_reward), 2)
    #         }
    #         if training:
    #             # Check if best reward
    #             if avg_reward > self._best_reward:
    #                 self._best_reward = avg_reward
    #                 self.agent.save()
    #         episode_logs.append(episode_log)

    #     # set _cur_obs to observation
    #     self._prev_obs = observation
    #     # self._prev_done = T.logical_or(observation.terminations, observation.truncations)

    #     return{
    #     'step_log': step_log,
    #     'episode_logs': episode_logs,
    #     }

    def train(self):
        """Trains On-Policy Agent following the Schedule."""
        self.initialize_run(context="train")
        # Set internal learn counter for timestep based learning

        # Initialize Rich Console
        console = Console()
        start_time = time.time()

        with Live(console=console, refresh_per_second=8, transient=True) as live:
            while not self.schedule.is_done(step=self._step, episodes=self._completed_episodes.sum().item()):
                
                step_result = self.step(training=True)
                self._step += self.env.num_envs
                self.add_to_normalizers()
                self.update_schedulers()

                if self.schedule.should_learn(
                    step=self._step,
                    episodes=self._completed_episodes.sum().item(),
                    last_learn_at=self._last_learn
                ):
                    learn_metrics = self.learn()
                    step_result['step_log'].update(learn_metrics)

                    # Update target networks
                    self.agent.soft_update_targets()

                    # Update normalizers
                    self.update_normalizers()

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