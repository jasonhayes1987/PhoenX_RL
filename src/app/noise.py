import torch as T
from torch.distributions import uniform, normal
import numpy as np
from torch_utils import get_device

class Noise:
    """
    Base class for noise processes.
    """

    def __init__(self, device=None):
        self.device = get_device(device)

    def __call__(self, shape):
        """
        Generate noise based on the specific implementation.

        Args:
            shape (tuple): Shape of the noise to generate.
        """
        pass

    def reset(self):
        """
        Reset the noise process (if applicable).
        """
        pass

    def get_config(self) -> dict:
        """
        Retrieve the configuration of the noise process.

        Returns:
            dict: Configuration details.
        """
        pass

    def clone(self):
        """
        Clone the noise process.

        Returns:
            Noise: A new instance of the same noise process.
        """
        pass

    @classmethod
    def create_instance(cls, noise_class_name: str, **kwargs) -> 'Noise':
        """
        Creates an instance of the requested noise class.

        Args:
            noise_class_name (str): Name of the noise class to instantiate.
            kwargs: Parameters for the noise class.

        Returns:
            Noise: An instance of the requested noise class.

        Raises:
            ValueError: If the noise class is not recognized.
        """
        noise_classes = {
            "Ornstein-Uhlenbeck": OUNoise,
            "OUNoise": OUNoise,
            "Normal": NormalNoise,
            "NormalNoise": NormalNoise,
            "Uniform": UniformNoise,
            "UniformNoise": UniformNoise,
        }

        if noise_class_name in noise_classes:
            return noise_classes[noise_class_name](**kwargs)
        else:
            raise ValueError(f"{noise_class_name} is not a recognized noise class")

class UniformNoise(Noise):
    """
    Uniform noise generator.
    """
    def __init__(self, shape, minval=0, maxval=1, device=None):
        super().__init__(device)
        self.shape = shape
        # self.device = T.device("cuda" if device == 'cuda' and T.cuda.is_available() else "cpu")
        self.minval = T.tensor(minval, device=self.device)
        self.maxval = T.tensor(maxval, device=self.device)
        
        self.noise_gen = uniform.Uniform(low=self.minval, high=self.maxval)

    def __call__(self) -> T.Tensor:
        """
        Generate uniform noise.

        Returns:
            T.Tensor: Generated noise.
        """
        return self.noise_gen.sample(self.shape)

    def get_config(self) -> dict:
        """
        Retrieve the configuration of the UniformNoise.

        Returns:
            dict: Configuration details.
        """
        return {
            'class_name': 'UniformNoise',
            'config': {
                'shape': self.shape,
                'minval': self.minval.item(),
                'maxval': self.maxval.item(),
                'device': self.device.type,
            }
        }
    
    def clone(self) -> 'UniformNoise':
        """
        Clone the UniformNoise instance.

        Returns:
            UniformNoise: A new instance with the same configuration.
        """
        return UniformNoise(self.shape, self.minval.item(), self.maxval.item(), self.device)

class NormalNoise(Noise):
    """
    Normal (Gaussian) noise generator.
    """
    def __init__(self, shape, mean=0.0, stddev=1.0, device=None):
        super().__init__(device)
        self.shape = shape
        # self.device = T.device("cuda" if device == 'cuda' and T.cuda.is_available() else "cpu")
        self.mean = T.tensor(mean, dtype=T.float32, device=self.device)
        self.stddev = T.tensor(stddev, dtype=T.float32, device=self.device)
        self.reset_noise_gen()

    def reset_noise_gen(self) -> None:
        """
        Reset the noise generator to the original mean and standard deviation.
        """
        self.noise_gen = normal.Normal(loc=self.mean, scale=self.stddev)

    def __call__(self) -> T.Tensor:
        """
        Generate normal noise.

        Returns:
            T.Tensor: Generated noise.
        """
        return self.noise_gen.sample(self.shape)

    def __getstate__(self):
        # Only the numpy arrays are serialized
        state = self.__dict__.copy()
        # Remove the noise generator since it can't be pickled
        del state['noise_gen']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate the noise generator after deserialization
        self.reset_noise_gen()

    def get_config(self) -> dict:
        """
        Retrieve the configuration of the NormalNoise.

        Returns:
            dict: Configuration details.
        """
        return {
            'class_name': 'NormalNoise',
            'config': {
                'shape': self.shape,
                'mean': self.mean.item(),
                'stddev': self.stddev.item(),
                'device': self.device.type,
            }
        }
    
    def clone(self) -> 'NormalNoise':
        """
        Clone the NormalNoise instance.

        Returns:
            NormalNoise: A new instance with the same configuration.
        """
        return NormalNoise(self.shape, self.mean.item(), self.stddev.item(), self.device)
    
class OUNoise(Noise):
    """
    Ornstein-Uhlenbeck noise process.

    Commonly used in reinforcement learning for exploration in continuous action spaces.
    """

    def __init__(self, shape: tuple, mean: float = 0.0, theta: float = 0.15, sigma: float = 0.2, dt: float = 1e-2, device=None):
        super().__init__(device)
        # self.device = T.device("cuda" if device == 'cuda' and T.cuda.is_available() else "cpu")
        self.shape = shape
        self.mean = T.tensor(mean, device=self.device)
        self.mu = T.ones(self.shape, device=self.device) * self.mean
        self.theta = T.tensor(theta, device=self.device)
        self.sigma = T.tensor(sigma, device=self.device)
        self.dt = T.tensor(dt, device=self.device)
        self.reset()

    def __call__(self) -> T.Tensor:
        """
        Generate Ornstein-Uhlenbeck noise.

        Returns:
            T.Tensor: Generated noise.
        """
        dx = self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * T.randn(self.shape, device=self.device)
        x = self.x_prev + dx
        self.x_prev = x
        return x

    def reset(self, mu: float = None) -> None:
        """
        Reset the noise process to its initial state.

        Args:
            mu (float, optional): New mean value. Defaults to the original mean.
        """
        # self.mu = T.ones(self.shape, device=self.device) * self.mean if mu is None else T.tensor(mu, device=self.device)
        # self.x_prev = T.ones(self.shape, device=self.device) * self.mu
        self.x_prev = T.ones(self.shape, device=self.device) * (mu if mu is not None else self.mean)

    def get_config(self) -> dict:
        """
        Retrieve the configuration of the OUNoise.

        Returns:
            dict: Configuration details.
        """
        return {
            'class_name': 'OUNoise',
            'config': {
                "shape": self.shape,
                "mean": self.mean.item(),
                "theta": self.theta.item(),
                "sigma": self.sigma.item(),
                "dt": self.dt.item(),
                'device': self.device.type,
            }
        }
        
    def clone(self) -> 'OUNoise':
        """
        Clone the OUNoise instance.

        Returns:
            OUNoise: A new instance with the same configuration.
        """
        return OUNoise(self.shape, self.mean.item(), self.theta.item(), self.sigma.item(), self.dt.item(), self.device)