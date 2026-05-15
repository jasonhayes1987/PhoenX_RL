from dataclasses import dataclass, field
import os
from typing import Literal
import logging
import numpy as np
import torch as T
import wandb
from moviepy.editor import ImageSequenceClip

from .env_wrapper import EnvWrapper, IsaacSimWrapper, GymnasiumWrapper, EnvPoolWrapper
from .rl_callbacks import WandbCallback
from .logging_config import get_logger


@dataclass
class Renderer:
    """
    Handles all rendering, video creation, and video logging.
    """
    render_freq: int = 0
    save_dir: str = "models/"
    fps: int = 30
    codec: str = "libx264"
    
    logger: logging.Logger = field(init=False)

    def __post_init__(self):
        self.logger = get_logger(self.__class__.__name__, level='INFO')

    def render_episode(
        self,
        trainer,  # OnPolicyTrainer — gives us get_action + env + agent
        episode: int,
        step: int,
        context: Literal["train", "test"] = "train",
        num_envs: int = 1,
        **kwargs
    ):
        """Universal render_episode — works with Gymnasium only for now."""
        env = trainer.env

        if isinstance(env, IsaacSimWrapper):
            raise ValueError(
                "Rendering episodes is not supported for IsaacSim environments. "
                "Test using one environment with render_mode='gui' instead."
            )

        self.logger.info(f"Rendering episode {episode} in {context} with kwargs: {kwargs}")

        # Clone a fresh env for rendering (respects **kwargs)
        render_env = env.clone(num_envs=num_envs, **kwargs)
        observation = render_env.reset()

        frames = []
        local_step = 0
        episode_reward = 0.0
        done = False

        while not done:
            local_step += 1
            # obs, goals, _ = trainer.extract_states_goals(states)
            obs_norm = trainer.normalize_observation(observation)

            actions = trainer.get_action(obs_norm.states, obs_norm.goals, context="test")
            # actions = env.format_actions(actions)

            # Step the render env
            next_observation = render_env.step(actions)

            # episode_reward += rewards[0].item() if isinstance(rewards, (T.Tensor, np.ndarray)) else rewards[0]
            episode_reward += float(next_observation.rewards[0])
            # frame = render_env.render_frame()
            frames.append(render_env.render_frame())

            observation = next_observation
            done = bool(observation.terminations[0].item()) or bool(observation.truncations[0].item())

        # Save video
        self._save_video(frames, episode, context)

        # Log to Wandb if callback exists
        video_path = os.path.join(self.save_dir, f"renders/{context}/episode_{episode}.mp4")
        if any(isinstance(cb, WandbCallback) for cb in trainer.callbacks):
            for cb in trainer.callbacks:
                if isinstance(cb, WandbCallback):
                    # caption = f"{context.capitalize()} render episode {episode}"
                    wandb.log({
                        f"{context}_video": wandb.Video(video_path, caption=f"{context} episode {episode}", format="mp4"),
                        f"render_episode_reward": episode_reward,
                        f"render_episode_length": local_step,
                    }, step=step)

        render_env.close()

    def _save_video(self, frames: list, episode: int, context: str):
        """Moved from utils.py — now inside Renderer."""

        if not isinstance(frames, np.ndarray):
            frames = np.array(frames)

        # Ensure directory exists
        video_dir = os.path.join(self.save_dir, f"renders/{context}")
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"episode_{episode}.mp4")

        clip = ImageSequenceClip(list(frames), fps=self.fps)
        clip.write_videofile(video_path, codec=self.codec, verbose=False, logger=None)
        self.logger.info(f"✅ Episode {episode} rendered → {video_path}")

    def should_render(self, episode: int) -> bool:
        """Checks if the renderer should render based on the current episode and render frequency."""
        if self.render_freq <= 0:
            return False
        return episode % self.render_freq == 0