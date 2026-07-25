"General utility functions"
import logging
import os
import torch as T
import numpy as np
# from .env_wrapper import EnvWrapper, GymnasiumWrapper, IsaacSimWrapper
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

# def render_video(frames: list, episode: int, save_dir: str, context: str = None) -> None:
#     """
#     Render a video from a list of frames and save it to a file.

#     Args:
#         frames (list): List of frames to render.
#         episode (int): Episode number for naming the output file.
#         save_dir (str): Directory to save the rendered video.
#         context (str): Context for the video (e.g., 'train', 'test').

#     Returns:
#         None
#     """
#     from moviepy.editor import ImageSequenceClip
#     print('rendering episode...')
#     if not isinstance(frames, np.ndarray):
#         frames = np.array(frames)
#     if context == 'train':
#         video_path = os.path.join(save_dir, f"renders/train/episode_{episode}.mp4")
#     elif context == 'test':
#         print('context set to test')
#         video_path = os.path.join(save_dir, f"renders/test/episode_{episode}.mp4")
#         print(f'video path:{video_path}')
#     else:
#         video_path = os.path.join(save_dir, f"renders/episode_{episode}.mp4")

#     # Ensure the directory exists
#     directory = os.path.dirname(video_path)
#     if not os.path.exists(directory):
#         os.makedirs(directory, exist_ok=True)

#     fps = 30
#     clip = ImageSequenceClip(list(frames), fps=fps)
#     clip.write_videofile(video_path, codec='libx264')
#     print('episode rendered')

# def build_env_wrapper_obj(config: dict) -> EnvWrapper:
#     """
#     Build an environment wrapper object based on the configuration.

#     Args:
#         config (dict): Configuration dictionary containing environment details.

#     Returns:
#         EnvWrapper: An instance of the appropriate environment wrapper.

#     Raises:
#         ValueError: If the wrapper type specified in the config is not recognized.
#     """
#     if config['type'] == "GymnasiumWrapper":
#         env = EnvSpec.from_json(config['env'])
#         return GymnasiumWrapper(env)
#     elif config['type'] == "IsaacSimWrapper":
#         pass
#     else:
#         raise ValueError(f"Environment wrapper {config['type']} not found")
    
def check_for_inf_or_NaN(value:T.Tensor, label:str):
    if T.any(T.isnan(value)):
        print(f'NAN found in {label}; {value}')
    elif T.any(T.isinf(value)):
        print(f'inf found in {label}; {value}')

def to_torch(value, device=None):
    if isinstance(value, np.ndarray) and value.dtype !=np.object_:
        if np.issubdtype(value.dtype, np.floating):
            return T.as_tensor(value, device=device, dtype=T.float32)
        if value.dtype == np.uint8:
            # Preserve uint8 (image observations) so buffers can store them
            # compactly; models cast/scale at their input boundary.
            return T.as_tensor(value, device=device, dtype=T.uint8)
        if np.issubdtype(value.dtype, np.integer):
            return T.as_tensor(value, device=device, dtype=T.int32)
        if np.issubdtype(value.dtype, np.bool_):
            return T.as_tensor(value, device=device, dtype=T.bool)
    if isinstance(value, dict):
        return {k: to_torch(v, device=device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(to_torch(v, device=device) for v in value)
    if isinstance(value, list):
        return [to_torch(v, device=device) for v in value]
    if isinstance(value, (np.generic, float, int, bool)):
        return T.as_tensor(value, device=device)
    return value

def to_numpy(value):
    if isinstance(value, T.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {k: to_numpy(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(to_numpy(v) for v in value)
    if isinstance(value, list):
        return [to_numpy(v) for v in value]
    return value

def summarize_tensor(x: T.Tensor | None, name: str) -> str:
    if x is None:
        return f"{name}=None"
    with T.no_grad():
        x = x.detach()
        flat = x.reshape(-1).float()
        numel = flat.numel()
        if numel == 0:
            return f"{name}[empty]"
        finite_mask = T.isfinite(flat)
        finite_frac = finite_mask.float().mean().item()
        if finite_mask.any():
            safe = flat[finite_mask]
            mean = safe.mean().item()
            std = safe.std(unbiased=False).item() if safe.numel() > 1 else 0.0
            min_v = safe.min().item()
            max_v = safe.max().item()
            p05, p50, p95 = T.quantile(
                safe, T.tensor([0.05, 0.50, 0.95], device=safe.device)
            ).tolist()
            return (
                f"{name}[shape={tuple(x.shape)}, finite={finite_frac:.3f}, "
                f"mean={mean:.4f}, std={std:.4f}, min={min_v:.4f}, "
                f"p05={p05:.4f}, p50={p50:.4f}, p95={p95:.4f}, max={max_v:.4f}]"
            )
        return f"{name}[shape={tuple(x.shape)}, finite=0.000, all_nonfinite=True]"

def count_nonfinite(x: T.Tensor | None) -> int:
    if x is None:
        return 0
    with T.no_grad():
        return (~T.isfinite(x.detach())).sum().item()