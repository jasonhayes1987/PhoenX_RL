import torch as T
import torch.nn as nn
from torch import optim
import numpy as np


def set_seed(seed: int):
    """
    Set the random seed for reproducibility in PyTorch and NumPy.

    Args:
        seed (int): The seed to set for all random number generators.
    """
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    np.random.seed(seed)

def VarianceScaling_(
    tensor: T.Tensor, 
    scale: float = 1.0, 
    mode: str = 'fan_in', 
    distribution: str = 'normal'
):
    """
    Apply variance scaling initialization to a tensor.

    Args:
        tensor (torch.Tensor): The tensor to initialize.
        scale (float): Scaling factor for the initialization. Default is 1.0.
        mode (str): Mode for scaling. Options are 'fan_in', 'fan_out', or 'fan_avg'. Default is 'fan_in'.
        distribution (str): Distribution to use for initialization. Options are 'normal', 'truncated_normal', or 'uniform'. Default is 'normal'.

    Raises:
        ValueError: If mode or distribution is not supported.
    """
    # Validate mode
    if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
        raise ValueError(f"Mode '{mode}' is not supported. Use 'fan_in', 'fan_out', or 'fan_avg'.")
    
    # Compute fan based on mode
    if mode == 'fan_in':
        fan = tensor.size(0)
    elif mode == 'fan_out':
        fan = tensor.size(1)
    else:  # mode == 'fan_avg'
        fan = (tensor.size(0) + tensor.size(1)) / 2

    val = T.sqrt(T.tensor(scale / fan))

    # Apply initialization based on distribution
    with T.no_grad():
        if distribution == 'normal':
            nn.init.normal_(tensor, mean=0.0, std=val)
        elif distribution == 'truncated_normal':
            nn.init.trunc_normal_(tensor, mean=0.0, std=val.item(), a=-2.0 * val.item(), b=2.0 * val.item())
        elif distribution == 'uniform':
            nn.init.uniform_(tensor, -val.item(), val.item())
        else:
            raise ValueError(
                f"Distribution '{distribution}' is not supported. Use 'normal', 'truncated_normal', or 'uniform'."
            )
    
def get_optimizer_by_name(name: str):
    """
    Retrieve an optimizer class by name.

    Args:
        name (str): Name of the optimizer (e.g., 'Adam', 'SGD').

    Returns:
        Optimizer class: The PyTorch optimizer class corresponding to the name.

    Raises:
        ValueError: If the optimizer name is not recognized.
    """
    opts = {
        "Adam": optim.Adam,
        "SGD": optim.SGD,
        "RMSprop": optim.RMSprop,
        "Adagrad": optim.Adagrad,
    }

    if name not in opts:
        raise ValueError(
            f'Optimizer "{name}" is not recognized. Available options: {list(opts.keys())}'
        )

    return opts[name]