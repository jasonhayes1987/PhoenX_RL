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
from .buffer import Buffer
from .renderer import Renderer
from .logging_config import get_logger, configure_logging

@dataclass
class Schedule:
    seed: int | None = None

@dataclass
class OnPolicySchedule(Schedule):
    # Training length
    unit: Literal["timestep", "episode"] = 'episode'
    num_units: int = 1000

    # learning Frequency
    learn_unit: Literal["timestep", "trajectory"] = 'trajectory'
    num_learn_units: int = 10

    # Algo specific (ie. PPO)
    batch_size: int|None = None
    learning_epochs: int|None = None

    # Set seed
    seed: int|None = None

# @dataclass
# class ActionOutput:
#     actions: T.Tensor

# @dataclass
# class StochasticActionOutput(ActionOutput):
#     mean: T.Tensor|None = None
#     mode: T.Tensor|None = None
#     std: T.Tensor|None = None
#     log_probs: T.Tensor|None = None
#     entropies: T.Tensor|None = None

class Trainer:
    def __init__(
        self,
        agent:Agent,
        env:EnvWrapper,
        buffer:Buffer,
        schedule:Schedule,
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
        self._cur_obs = None
        self._best_reward = None
        self._completed_episodes = None
        self._episode_scores = None
        self._score_history = None

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
                    model.logger.info(f"Set {name} to train mode")
                elif context == "test":
                    model.eval()
                    model.logger.info(f"Set {name} to eval mode")


        # Set target models to eval mode
        for name in ['target_policy', 'target_critic', 'target_critic_a', 'target_critic_b']:
            model = getattr(self.agent, name, None)
            if model:
                model.eval()
                model.logger.info(f"Set {name} to eval mode")
        
        # Set internal attributes
        seed = self.schedule.seed if self.schedule.seed else T.randint(0, 1000000)
        set_seed(seed)
        observation = self.env.reset()

        # Warmup normalizers if exist
        if self.agent.state_normalizer or self.agent.goal_normalizer:
            if context == "train":
                for _ in range(self.agent.state_normalizer.warmup_steps):
                    observation = self.env.sample_observation()
                    if self.agent.state_normalizer:
                        self.agent.state_normalizer.train()
                        self.agent.state_normalizer.add(observation.current_states.to(device=self.agent.state_normalizer.device))
                    if self.agent.goal_normalizer:
                        self.agent.goal_normalizer.train()
                        # Concatenate current goals and achieved goals into 1 tensor
                        goals = T.cat([observation.current_goals, observation.current_ach_goals], dim=0).to(device=self.agent.goal_normalizer.device)
                        self.agent.goal_normalizer.add(goals)
                    if self.agent.advantage_normalizer:
                        if context == "train":
                            self.agent.advantage_normalizer.train()
                # Update normalizers
                if self.agent.state_normalizer:
                    self.agent.state_normalizer.update()
                if self.agent.goal_normalizer:
                    self.agent.goal_normalizer.update()
                # Reset environments
                observation = self.env.reset()
            else:
                if self.agent.state_normalizer:
                    self.agent.state_normalizer.eval()
                if self.agent.goal_normalizer:
                    self.agent.goal_normalizer.eval()
                if self.agent.advantage_normalizer:
                    self.agent.advantage_normalizer.eval()

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
        self._cur_obs = observation
        self._best_reward = -T.inf
        self._completed_episodes = T.zeros(self.env.num_envs, dtype=T.int32, device=self.agent.device)
        self._episode_scores = T.zeros(self.env.num_envs, dtype=T.float32, device=self.agent.device)
        self._score_history = deque(maxlen=100)
        self._render_counter = 0

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

    @abstractmethod
    def get_action(self, states:np.ndarray|T.Tensor, goals:np.ndarray|T.Tensor|None, context: str = 'train')->T.Tensor:
        """Get an action from the agent."""
        raise NotImplementedError("Subclasses must implement get_action.")

    @abstractmethod
    def step(self):
        """Steps the trainer."""
        raise NotImplementedError("Subclasses must implement step.")

    @abstractmethod
    def train(self):
        """Trains the Agent."""
        raise NotImplementedError("Subclasses must implement train.")

class OnPolicyTrainer(Trainer):
    def __init__(
        self,
        agent: Agent,
        env: EnvWrapper,
        buffer: Buffer,
        schedule: OnPolicySchedule,
        renderer: Renderer | None = None,
        callbacks: list[Callback] | None = None,
        log_level: str = 'INFO',
    ):
        super().__init__(agent, env, buffer, schedule, renderer, callbacks, log_level)
        self._learn_counter = 0

    def should_learn(self)->bool:
        """Checks if the agent should learn based on the schedule."""
        if self.schedule.learn_unit not in ['timestep', 'trajectory']:
            raise ValueError(f"Invalid learn unit: {self.schedule.learn_unit}")
        elif self.schedule.learn_unit == 'timestep':
            self._learn_counter += self.env.num_envs
            if self._learn_counter >= self.schedule.num_learn_units:
                self._learn_counter = 0
                return True
        else: # self.schedule.learn_unit == 'trajectory'
            if self._completed_episodes.sum() >= self.schedule.num_learn_units * (self._learn_counter + 1):
                self._learn_counter += 1
                return True
        return False

    def learn(self)->dict:
        """Calls Agent.learn() passing samples from the buffer.
        Will also pass batch_size and learning_epochs from the schedule if they are set.
        
        Returns:
            dict: A dictionary containing the learn metrics.
        """
        if self.schedule.batch_size is not None and self.schedule.learning_epochs is not None:
            learn_metrics = self.agent.learn(self._step, self.buffer.sample(), self.schedule.batch_size, self.schedule.learning_epochs)
        else:
            learn_metrics = self.agent.learn(self._step, self.buffer.sample())
        return learn_metrics

    def get_action(
        self,
        states: np.ndarray | T.Tensor,
        goals: np.ndarray | T.Tensor | None = None,
        context: str = 'train'
    ) -> T.Tensor:
        """
        Select an action based on the current policy.
        Returns actions that are already scaled to the environment's action space.
        """

        if context == 'test':
            with T.no_grad():
                dist = self.agent.policy(states, goals)
        else:
            dist = self.agent.policy(states, goals)
        
        if context == 'train':
            actions = dist.sample()
        elif context == 'test':
            actions = self.agent.policy.get_mean_actions(dist)
        else:
            raise ValueError(f"Invalid context: {context}")

        return actions

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
        obs_norm, goals_norm, ach_goals_norm = self.normalize_inputs(self._cur_obs.current_states, self._cur_obs.current_goals, self._cur_obs.current_ach_goals)

        actions = self.get_action(obs_norm, goals_norm, context='train' if training else 'test')
        observation = self.env.step(actions)

        if training:
            # Update normalizers
            if self.agent.state_normalizer:
                self.agent.state_normalizer.add(observation.transition_states.to(device=self.agent.state_normalizer.device))
            if self.agent.goal_normalizer:
                next_goals = T.cat([observation.transition_goals, observation.transition_ach_goals], dim=0).to(device=self.agent.goal_normalizer.device)
                self.agent.goal_normalizer.add(next_goals)

            # Step schedulers
            if self.agent.entropy_schedule:
                self.agent.entropy_schedule.step()
            if getattr(self.agent.policy, 'temperature_schedule', None):
                self.agent.policy.temperature_schedule.step()

        # next_obs_norm, next_goals_norm, next_ach_goals_norm = self.normalize_inputs(observation.transition_states, observation.transition_goals, observation.transition_ach_goals)
        
        # Add normalized transitions to the buffer
        self.buffer.add(
            states=self._cur_obs.current_states,
            actions=actions,
            rewards=observation.rewards,
            next_states=observation.transition_states,
            terminations=observation.terminations,
            truncations=observation.truncations,
            state_achieved_goals=self._cur_obs.current_ach_goals,
            next_state_achieved_goals=observation.transition_ach_goals,
            desired_goals=self._cur_obs.current_goals,
        )

        self._episode_scores += observation.rewards.flatten()

        # Add step metrics to step log
        step_log.update({
            'step_reward': observation.rewards.mean().item()
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
        self._cur_obs = observation

        return{
        'step_log': step_log,
        'episode_logs': episode_logs,
    }

    def train(self):
        """Trains On-Policy Agent following the Schedule."""
        self.initialize_run(context="train")
        # Set internal learn counter for timestep based learning
        self._learn_counter = 0

        while True:
            if self.schedule.unit == 'timestep':
                if self._step >= self.schedule.num_units:
                    break
            elif self.schedule.unit == 'episode':
                if self._completed_episodes.sum() >= self.schedule.num_units:
                    break
            else:
                raise ValueError(f"Invalid unit: {self.schedule.unit}")

            step_result =self.step(training=True)
            self._step += self.env.num_envs

            if self.should_learn():
                learn_metrics = self.learn()
                step_result['step_log'].update(learn_metrics)

            # Update normalizers
            if self.agent.state_normalizer:
                self.agent.state_normalizer.update()
            if self.agent.goal_normalizer:
                self.agent.goal_normalizer.update()

            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_train_step_end(self._step, step_result['step_log'])

            
            for episode_log in step_result['episode_logs']:
                self.logger.info(f"Training Episode {episode_log['episode']}: Reward {episode_log['episode_reward']}, Avg Reward {episode_log['avg_reward']}")
                # Reset episode score to 0
                self._episode_scores[episode_log['env']] = 0
                if self.renderer and self.renderer.should_render(episode_log['episode']):
                    # Set normalizers to eval mode
                    if self.agent.state_normalizer:
                        self.agent.state_normalizer.eval()
                    if self.agent.goal_normalizer:
                        self.agent.goal_normalizer.eval()
                    if self.agent.advantage_normalizer:
                        self.agent.advantage_normalizer.eval()
                    self.renderer.render_episode(self, episode_log['episode'], self._step, context='train', seed=T.randint(high=1000000, size=(1,)).item(), render_mode='rgb_array')
                    # Set normalizers to train mode
                    if self.agent.state_normalizer:
                        self.agent.state_normalizer.train()
                    if self.agent.goal_normalizer:
                        self.agent.goal_normalizer.train()
                    if self.agent.advantage_normalizer:
                        self.agent.advantage_normalizer.train()
                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_train_epoch_end(self._step, episode_log)

        if self.callbacks:
            for episode_log in step_result['episode_logs']:
                for callback in self.callbacks:
                    callback.on_train_end(episode_log)

        self.env.close()