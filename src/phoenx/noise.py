"""Exploration noise processes for continuous-action agents.

Off-policy actors (DDPG, TD3) add sampled noise to deterministic actions
during training. ``Noise`` is the abstract interface; ``UniformNoise`` and
``NormalNoise`` are the concrete generators. Instances are built by name
through ``Noise.create_instance``, and ``get_config`` / ``clone`` support
checkpointing and device moves.
"""

from typing import Optional
from abc import ABC, abstractmethod
import torch as T
from torch.distributions import uniform, normal
import numpy as np
from .torch_utils import get_device

class Noise(ABC):
    """Abstract exploration-noise process.

    Subclasses implement ``__call__`` to sample a tensor of a requested shape,
    plus ``get_config`` and ``clone`` for serialization and device moves.
    """

    def __init__(self, device=None):
        """Bind the noise process to a torch device.

        Args:
            device (torch.device | str | None): Device for sampled tensors;
                ``None`` resolves through ``torch_utils.get_device``.
        """
        self.device = get_device(device)

    @abstractmethod
    def __call__(self, shape: tuple) -> T.Tensor:
        """Generate noise based on the specific implementation.

        Args:
            shape: Shape of the noise to generate.
        """
        raise NotImplementedError("Subclasses must implement reset.")

    @abstractmethod
    def get_config(self) -> dict:
        """Retrieve the configuration of the noise process.

        Returns:
            Configuration details.
        """
        raise NotImplementedError("Subclasses must implement get_config.")

    @abstractmethod
    def clone(self, device: Optional[str | T.device] = None) -> 'Noise':
        """Clone the noise process.

        Args:
            device: Optional device for the clone; ``None`` keeps the current
                device.

        Returns:
            A new instance of the same noise process.
        """
        raise NotImplementedError("Subclasses must implement clone.")

    @classmethod
    def create_instance(cls, noise_type: str, **kwargs) -> 'Noise':
        """Creates an instance of the requested noise class.

        Args:
            noise_type: Name of the noise class to instantiate.
            kwargs (dict): Constructor keyword arguments for the noise class.

        Returns:
            An instance of the requested noise class.

        Raises:
            ValueError: If the noise class is not recognized.
        """
        noise_classes = {
            # "Ornstein-Uhlenbeck": OUNoise,
            # "OUNoise": OUNoise,
            "Normal": NormalNoise,
            "NormalNoise": NormalNoise,
            "Uniform": UniformNoise,
            "UniformNoise": UniformNoise,
        }

        if noise_type in noise_classes:
            return noise_classes[noise_type](**kwargs)
        else:
            raise ValueError(f"{noise_type} is not a recognized noise class")

class UniformNoise(Noise):
    """Uniform exploration noise over ``[minval, maxval]``.

    Each ``__call__`` draws an independent sample from
    ``torch.distributions.Uniform``. The default sample shape is fixed at
    construction; callers may override it per call.
    """
    def __init__(self, shape, minval=0, maxval=1, device=None):
        """Create a uniform noise generator.

        Args:
            shape (tuple): Default sample shape used when ``__call__`` is given
                no shape.
            minval (float): Inclusive lower bound of the uniform interval.
            maxval (float): Exclusive upper bound of the uniform interval.
            device (torch.device | str | None): Device for sampled tensors;
                ``None`` resolves through ``torch_utils.get_device``.
        """
        super().__init__(device)
        self.shape = shape
        # self.device = T.device("cuda" if device == 'cuda' and T.cuda.is_available() else "cpu")
        self.minval = T.tensor(minval, device=self.device)
        self.maxval = T.tensor(maxval, device=self.device)
        
        self.noise_gen = uniform.Uniform(low=self.minval, high=self.maxval)

    def __call__(self, shape: tuple=None) -> T.Tensor:
        """Generate uniform noise.

        Args:
            shape: Sample shape; ``None`` uses the shape fixed at construction.

        Returns:
            Sampled noise tensor.
        """
        if shape is None:
            shape = self.shape
        return self.noise_gen.sample(shape)

    def get_config(self) -> dict:
        """Retrieve the configuration of the UniformNoise.

        Returns:
            Configuration details.
        """
        return {
            'type': 'UniformNoise',
            'config': {
                'shape': self.shape,
                'minval': self.minval.item(),
                'maxval': self.maxval.item(),
                'device': self.device.type,
            }
        }
    
    def clone(self, device: Optional[str | T.device] = None) -> 'UniformNoise':
        """Clone the UniformNoise instance.

        Args:
            device: Optional device for the clone; ``None`` keeps the current
                device.

        Returns:
            A new instance with the same configuration.
        """
        if device:
            device = get_device(device)
        else:
            device = self.device

        return UniformNoise(self.shape, self.minval.item(), self.maxval.item(), device)

class NormalNoise(Noise):
    """Independent Gaussian exploration noise.

    Samples from ``torch.distributions.Normal`` with fixed mean and standard
    deviation. The distribution is rebuilt after unpickling via
    ``__setstate__``.
    """
    def __init__(self, mean=0.0, stddev=1.0, device=None):
        """Create a Gaussian noise generator.

        Args:
            mean (float): Mean of the normal distribution.
            stddev (float): Standard deviation of the normal distribution.
            device (torch.device | str | None): Device for sampled tensors;
                ``None`` resolves through ``torch_utils.get_device``.
        """
        super().__init__(device)
        self.mean = T.tensor(mean, dtype=T.float32, device=self.device)
        self.stddev = T.tensor(stddev, dtype=T.float32, device=self.device)
        self.reset_noise_gen()

    def reset_noise_gen(self) -> None:
        """Rebuild the underlying ``Normal`` from the stored mean and stddev."""
        self.noise_gen = normal.Normal(loc=self.mean, scale=self.stddev)

    def __call__(self, shape: Optional[tuple[int, ...]]=(1,1)) -> T.Tensor:
        """Generate normal noise.

        Args:
            shape: Sample shape (e.g. ``(batch_size, action_dim)``). Defaults
                to ``(1, 1)``.

        Returns:
            Sampled noise tensor.
        """
        if isinstance(shape, (np.ndarray, T.Tensor)):
           shape = tuple(shape)
        return self.noise_gen.sample(shape)

    def __getstate__(self):
        """Return picklable state with the unpicklable ``noise_gen`` removed."""
        # Only the numpy arrays are serialized
        state = self.__dict__.copy()
        # Remove the noise generator since it can't be pickled
        del state['noise_gen']
        return state

    def __setstate__(self, state):
        """Restore instance state and recreate the noise generator.

        Args:
            state (dict): Instance ``__dict__`` without ``noise_gen``.
        """
        self.__dict__.update(state)
        # Recreate the noise generator after deserialization
        self.reset_noise_gen()

    def get_config(self) -> dict:
        """Retrieve the configuration of the NormalNoise.

        Returns:
            Configuration details.
        """
        return {
            'type': 'NormalNoise',
            'config': {
                'mean': self.mean.item(),
                'stddev': self.stddev.item(),
                'device': self.device.type,
            }
        }
    
    def clone(self, device: Optional[str | T.device] = None) -> 'NormalNoise':
        """Clone the NormalNoise instance.

        Args:
            device: Optional device for the clone; ``None`` keeps the current
                device.

        Returns:
            A new instance with the same configuration.
        """
        if device:
            device = get_device(device)
        else:
            device = self.device

        return NormalNoise(self.mean.item(), self.stddev.item(), device)
    
# class OUNoise(Noise):
#     """
#     Ornstein-Uhlenbeck noise process.

#     Commonly used in reinforcement learning for exploration in continuous action spaces.
#     """

#     def __init__(self, shape: tuple, mean: float = 0.0, theta: float = 0.15, sigma: float = 0.2, dt: float = 1e-2, device=None):
#         super().__init__(device)
#         # self.device = T.device("cuda" if device == 'cuda' and T.cuda.is_available() else "cpu")
#         self.shape = shape
#         self.mean = T.tensor(mean, device=self.device)
#         self.mu = T.ones(self.shape, device=self.device) * self.mean
#         self.theta = T.tensor(theta, device=self.device)
#         self.sigma = T.tensor(sigma, device=self.device)
#         self.dt = T.tensor(dt, device=self.device)
#         self.reset()

#     def __call__(self, shape: tuple=None) -> T.Tensor:
#         """
#         Generate Ornstein-Uhlenbeck noise.

#         Returns:
#             T.Tensor: Generated noise.
#         """
#         if shape is None:
#             shape = self.shape
#         dx = self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * T.randn(shape, device=self.device)
#         x = self.x_prev + dx
#         self.x_prev = x
#         return x

#     def reset(self) -> None:
#         """
#         Reset the noise process to its initial state.
#         """
#         # Reset the noise process to its initial state
#         self.x_prev = T.ones(self.shape, device=self.device) * self.mean

#     def get_config(self) -> dict:
#         """
#         Retrieve the configuration of the OUNoise.

#         Returns:
#             dict: Configuration details.
#         """
#         return {
#             'type': 'OUNoise',
#             'config': {
#                 "shape": self.shape,
#                 "mean": self.mean.item(),
#                 "theta": self.theta.item(),
#                 "sigma": self.sigma.item(),
#                 "dt": self.dt.item(),
#                 'device': self.device.type,
#             }
#         }
        
#     def clone(self, device: Optional[str | T.device] = None) -> 'OUNoise':
#         """
#         Clone the OUNoise instance.

#         Returns:
#             OUNoise: A new instance with the same configuration.
#         """
#         if device:
#             device = get_device(device)
#         else:
#             device = self.device

#         return OUNoise(self.shape, self.mean.item(), self.theta.item(), self.sigma.item(), self.dt.item(), device)
