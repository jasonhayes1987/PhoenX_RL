import torch as T
import torch.nn as nn
from torch import optim
import numpy as np
from typing import Optional


def move_to_device(obj, device: T.device | str, visited=None) -> object:
    """
    Recursively move all tensors and custom objects with device attributes to the specified device.
    
    Args:
        obj: Object to move (can be tensor, module, custom object, list, dict, etc.).
        device (torch.device): Target device (e.g., 'cuda', 'cpu').
        visited: Set of object ids already visited (for cycle detection).
    
    Returns:
        object: Object with all tensors moved to the target device.
    """
    # Initialize visited set if None (first call)
    if visited is None:
        visited = set()
    
    # Handle None and primitive types directly
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj

    # Check for cycles to avoid infinite recursion
    obj_id = id(obj)
    if obj_id in visited:
        return obj
    visited.add(obj_id)
    
    # Convert device to string for comparison
    device_str = str(device).split(':')[0]  # Get 'cuda' or 'cpu' part
    
    # Handle PyTorch types
    if isinstance(obj, T.nn.Module):
        return obj.to(device)
    elif isinstance(obj, T.Tensor):
        return obj.to(device)
    
    # Safe check for .to() method without using hasattr (which can cause recursion)
    to_method = getattr(obj, 'to', None)
    if to_method is not None and callable(to_method) and not isinstance(obj, type):
        try:
            return obj.to(device)
        except (TypeError, ValueError, AttributeError):
            # If .to() method exists but doesn't work with our device, continue with other approaches
            pass
    
    # Handle container types    
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # Special case for device entries
            if k == 'device' and isinstance(v, (str, T.device)):
                result[k] = device_str
            else:
                result[k] = move_to_device(v, device, visited)
        return result
    
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(v, device, visited) for v in obj)
    
    # Handle custom objects with __dict__
    if hasattr(obj, '__dict__'):
        # Set device attribute if it exists
        if hasattr(obj, 'device'):
            try:
                setattr(obj, 'device', device if isinstance(getattr(obj, 'device'), T.device) else device_str)
            except (AttributeError, TypeError):
                pass
        
        # Handle attributes that might contain device references
        safe_dict = {}
        try:
            safe_dict = obj.__dict__.copy()  # Make a copy to avoid modification during iteration
        except (AttributeError, TypeError):
            pass
            
        for attr_name, attr_value in safe_dict.items():
            # Skip certain attributes to avoid recursion issues
            if attr_name.startswith('_') and attr_name != '_config':
                continue
                
            # Skip known problematic attribute types
            if attr_name in ('__class__', '__weakref__', '__module__'):
                continue
                
            try:
                # Handle tensor and module attributes
                if isinstance(attr_value, (T.Tensor, T.nn.Module)):
                    setattr(obj, attr_name, attr_value.to(device))
                # Handle device attribute
                elif attr_name == 'device' and isinstance(attr_value, (T.device, str)):
                    # setattr(obj, attr_name, device if isinstance(attr_value, T.device) else device_str)
                    setattr(obj, attr_name, get_device(device_str))
                # Recursively process containers and custom objects
                elif isinstance(attr_value, (dict, list, tuple)) or (hasattr(attr_value, '__dict__') and not isinstance(attr_value, type)):
                    setattr(obj, attr_name, move_to_device(attr_value, device, visited))
            except (AttributeError, TypeError):
                # Skip attributes that can't be set or accessed
                continue
        
        # Special handling for _config dictionary if it exists
        if hasattr(obj, '_config') and isinstance(obj._config, dict):
            try:
                config = obj._config.copy()
                # Update the device in the main config
                if 'device' in config:
                    config['device'] = device_str
                # Update devices in any model configurations
                for model_key in ['actor_model', 'critic_model', 'value_model', 'policy_model']:
                    if model_key in config and isinstance(config[model_key], dict):
                        if 'device' in config[model_key]:
                            config[model_key]['device'] = device_str
                obj._config = config
            except (AttributeError, TypeError):
                pass
    
    return obj

def verify_device(obj, expected_device: str | T.device, verbose=False, indent=0, visited=None):
    """
    Recursively verifies that all tensors and objects with device attributes are on the expected device.
    
    Args:
        obj: Object to check (can be tensor, module, custom object, list, dict, etc.).
        expected_device (torch.device or str): Expected device ('cuda' or 'cpu').
        verbose (bool): Whether to print detailed information about each component.
        indent (int): Indentation level for verbose output.
        visited: Set of object ids already visited (for cycle detection).
    
    Returns:
        dict: Statistics about components on the correct and incorrect devices.
    """
    # Initialize visited set if None (first call)
    if visited is None:
        visited = set()
    
    # Initialize statistics
    stats = {
        'correct': 0,
        'incorrect': 0,
        'incorrect_devices': set(),
        'total': 0
    }
    
    # Convert expected_device to string for comparison
    if isinstance(expected_device, T.device):
        expected_device = expected_device.type.split(':')[0]  # 'cuda' or 'cpu'
    else:
        expected_device = expected_device.split(':')[0]  # 'cuda' or 'cpu'
    
    # Check for None or primitive types
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return stats
    
    # Check for cycles
    obj_id = id(obj)
    if obj_id in visited:
        return stats
    visited.add(obj_id)
    
    # Check for tensors
    if isinstance(obj, T.Tensor):
        actual_device = obj.device.type
        stats['total'] += 1
        if actual_device == expected_device:
            stats['correct'] += 1
            if verbose:
                print(' ' * indent + f"✓ Tensor: {actual_device}")
        else:
            stats['incorrect'] += 1
            stats['incorrect_devices'].add(actual_device)
            if verbose:
                print(' ' * indent + f"✗ Tensor on {actual_device}, expected {expected_device}")
    
    # Check for nn.Modules
    elif isinstance(obj, T.nn.Module):
        name = obj.__class__.__name__
        if verbose:
            print(' ' * indent + f"Module: {name}")
            
        # Check parameters and buffers
        for param_name, param in obj.named_parameters():
            actual_device = param.device.type
            stats['total'] += 1
            if actual_device == expected_device:
                stats['correct'] += 1
                if verbose:
                    print(' ' * (indent+2) + f"✓ Parameter {param_name}: {actual_device}")
            else:
                stats['incorrect'] += 1
                stats['incorrect_devices'].add(actual_device)
                if verbose:
                    print(' ' * (indent+2) + f"✗ Parameter {param_name} on {actual_device}, expected {expected_device}")
        
        for buffer_name, buffer in obj.named_buffers():
            actual_device = buffer.device.type
            stats['total'] += 1
            if actual_device == expected_device:
                stats['correct'] += 1
                if verbose:
                    print(' ' * (indent+2) + f"✓ Buffer {buffer_name}: {actual_device}")
            else:
                stats['incorrect'] += 1
                stats['incorrect_devices'].add(actual_device)
                if verbose:
                    print(' ' * (indent+2) + f"✗ Buffer {buffer_name} on {actual_device}, expected {expected_device}")
    
    # Check objects with device attribute
    elif hasattr(obj, 'device'):
        try:
            if isinstance(obj.device, T.device):
                actual_device = obj.device.type
            else:
                actual_device = str(obj.device).split(':')[0]
                
            stats['total'] += 1
            if actual_device == expected_device:
                stats['correct'] += 1
                if verbose:
                    print(' ' * indent + f"✓ Object {obj.__class__.__name__}: device={actual_device}")
            else:
                stats['incorrect'] += 1
                stats['incorrect_devices'].add(actual_device)
                if verbose:
                    print(' ' * indent + f"✗ Object {obj.__class__.__name__}: device={actual_device}, expected {expected_device}")
        except (AttributeError, TypeError):
            pass  # Skip if device can't be accessed
    
    # Recursively check dictionaries
    if isinstance(obj, dict):
        if verbose:
            print(' ' * indent + f"Dict: {len(obj)} items")
        for k, v in obj.items():
            if verbose:
                print(' ' * (indent+2) + f"Key: {k}")
            child_stats = verify_device(v, expected_device, verbose, indent+4, visited)
            for key in stats:
                if key == 'incorrect_devices' and isinstance(child_stats[key], set):
                    stats[key].update(child_stats[key])
                elif isinstance(stats[key], (int, float)):
                    stats[key] += child_stats[key]
    
    # Recursively check lists and tuples
    elif isinstance(obj, (list, tuple)):
        if verbose:
            print(' ' * indent + f"{type(obj).__name__}: {len(obj)} items")
        for i, item in enumerate(obj):
            if verbose and i < 5:  # Limit verbosity for large lists
                print(' ' * (indent+2) + f"Item {i}:")
            child_stats = verify_device(item, expected_device, verbose if i < 5 else False, indent+4, visited)
            for key in stats:
                if key == 'incorrect_devices' and isinstance(child_stats[key], set):
                    stats[key].update(child_stats[key])
                elif isinstance(stats[key], (int, float)):
                    stats[key] += child_stats[key]
    
    # Recursively check objects with __dict__
    elif hasattr(obj, '__dict__'):
        class_name = obj.__class__.__name__
        if verbose:
            print(' ' * indent + f"Object: {class_name}")
        
        # Skip certain objects to avoid infinite recursion
        if class_name in ('type', 'module', 'function', 'method', 'builtin_function_or_method'):
            return stats
        
        try:
            for attr_name, attr_value in obj.__dict__.items():
                # Skip private attributes
                if attr_name.startswith('_') and attr_name not in ('_parameters', '_buffers', '_modules', '_config'):
                    continue
                
                if verbose:
                    print(' ' * (indent+2) + f"Attribute: {attr_name}")
                child_stats = verify_device(attr_value, expected_device, verbose, indent+4, visited)
                for key in stats:
                    if key == 'incorrect_devices' and isinstance(child_stats[key], set):
                        stats[key].update(child_stats[key])
                    elif isinstance(stats[key], (int, float)):
                        stats[key] += child_stats[key]
        except (AttributeError, TypeError):
            # Skip if __dict__ can't be accessed
            pass
    
    return stats

def get_device(device_spec: Optional[str | T.device] = None) -> T.device:
    """
    Convert any valid device specification to a torch.device object.
    
    Args:
        device_spec: Can be a string ('cuda', 'cpu'), a torch.device object, 
                    or None (defaults to 'cuda' if available, else 'cpu')
                    
    Returns:
        torch.device: The corresponding device object
    """

    if device_spec is None:
        return T.device('cuda' if T.cuda.is_available() else 'cpu')
    elif isinstance(device_spec, str):
        return T.device('cuda' if device_spec == 'cuda' and T.cuda.is_available() else 'cpu')
    elif isinstance(device_spec, T.device):
        return device_spec
    else:
        raise ValueError(f"Unsupported device specification: {device_spec}")

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

