"General utility functions"
import os
import torch as T
import numpy as np
from moviepy.editor import ImageSequenceClip
from env_wrapper import GymnasiumWrapper, IsaacSimWrapper


def flatten_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary.

    Parameters:
    - d: The dictionary to flatten.
    - parent_key: The base key to use for the current level of recursion.
    - sep: The separator between nested keys.

    Returns:
    A flattened dictionary with concatenated keys.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def calculate_gae(rewards: T.tensor,
                values: T.tensor,
                next_values: T.tensor,
                dones: T.tensor,
                gamma = 0.99,
                lambda_ = 0.95
                ):

    deltas = rewards + gamma * next_values - values
    deltas = deltas.flatten()


    advantages = T.zeros_like(rewards, dtype=T.float, device=rewards.device)
    gae = 0
    # advantages = [0]
    # Iterate in reverse to calculate GAE
    for i in reversed(range(len(rewards))):
        gae = deltas[i] + gamma * lambda_ * gae * (1-dones[i])
        # print(f'gae: {gae}')
        # print(f'gae shape: {gae.shape}')
        # advantages.append(gae)
        advantages[i] = gae
    # advantages.reverse()
    # advantages = advantages[:-1]
    # advantages = T.tensor(advantages, dtype=T.float, device=self.value_function.device).unsqueeze(1)

    return advantages

def render_video(frames, episode, save_dir, context:str=None):
        print('rendering episode...')
        if not isinstance(frames, np.ndarray):
            frames = np.array(frames)
        if context == 'train':
            video_path = os.path.join(save_dir, f"renders/train/episode_{episode}.mp4")
        elif context == 'test':
            print('context set to test')
            video_path = os.path.join(save_dir, f"renders/test/episode_{episode}.mp4")
            print(f'video path:{video_path}')
        else:
            video_path = os.path.join(save_dir, f"renders/episode_{episode}.mp4")

        # Ensure the directory exists
        directory = os.path.dirname(video_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        fps = 30
        clip = ImageSequenceClip(list(frames), fps=fps)
        clip.write_videofile(video_path, codec='libx264')
        print('episode rendered')

def build_env_wrapper_obj(config:dict):
    if config['type'] == "GymnasiumWrapper":
        return GymnasiumWrapper(config['env'])
    elif config['type'] == "IsaacSimWrapper":
        pass
    else:
        raise ValueError(f"Environment wrapper {config['type']} not found")