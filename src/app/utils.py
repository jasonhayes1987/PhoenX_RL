"General utility functions"
import os
import torch as T
import numpy as np
from moviepy.editor import ImageSequenceClip
from env_wrapper import EnvWrapper, GymnasiumWrapper, IsaacSimWrapper
from gymnasium.envs.registration import EnvSpec


def flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
    """
    Flatten a nested dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key to use for the current level of recursion (default is '').
        sep (str): The separator between nested keys (default is '_').

    Returns:
        dict: A flattened dictionary with concatenated keys.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def render_video(frames: list, episode: int, save_dir: str, context: str = None) -> None:
    """
    Render a video from a list of frames and save it to a file.

    Args:
        frames (list): List of frames to render.
        episode (int): Episode number for naming the output file.
        save_dir (str): Directory to save the rendered video.
        context (str): Context for the video (e.g., 'train', 'test').

    Returns:
        None
    """
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

def build_env_wrapper_obj(config: dict) -> EnvWrapper:
    """
    Build an environment wrapper object based on the configuration.

    Args:
        config (dict): Configuration dictionary containing environment details.

    Returns:
        EnvWrapper: An instance of the appropriate environment wrapper.

    Raises:
        ValueError: If the wrapper type specified in the config is not recognized.
    """
    if config['type'] == "GymnasiumWrapper":
        env = EnvSpec.from_json(config['env'])
        return GymnasiumWrapper(env)
    elif config['type'] == "IsaacSimWrapper":
        pass
    else:
        raise ValueError(f"Environment wrapper {config['type']} not found")
    
def check_for_inf_or_NaN(value:T.Tensor, label:str):
    if T.any(T.isnan(value)):
        print(f'NAN found in {label}; {value}')
    elif T.any(T.isinf(value)):
        print(f'inf found in {label}; {value}')