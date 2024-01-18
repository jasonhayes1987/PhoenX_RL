"""This module provides helper functions for configuring and using TensorFlow."""

from tensorflow.keras import optimizers


def get_optimizer_by_name(name: str):
    """Creates and returns an optimizer object by its string name.

    Args:
        name (str): The name of the optimizer.

    Returns:
        An instance of the requested optimizer.

    Raises:
        ValueError: If the optimizer name is not recognized.
    """
    opts = {
        "adam": optimizers.Adam,
        "sgd": optimizers.SGD,
        "rmsprop": optimizers.RMSprop,
        "adagrad": optimizers.Adagrad,
        # Add more optimizers as needed
    }

    if name.lower() not in opts:
        raise ValueError(
            f'Optimizer "{name}" is not recognized. Available options: {list(opts.keys())}'
        )

    return opts[name.lower()]()
