import torch as T
from torch.distributions import uniform, normal
import numpy as np
import env_wrapper


class Noise:
    """Base class for noise processes."""

    def __init__(self):
        pass

    def __call__(self, shape):
        pass

    def reset(self):
        pass

    def get_config(self):
        pass

    def clone(self):
        pass

    @classmethod
    def create_instance(cls, noise_class_name, **kwargs):
        """Creates an instance of the requested noise class.

        Args:
            noise_class_name (str): The name of the noise class.

        Returns:
            Noise: An instance of the requested noise class.
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
    def __init__(self, shape, minval=0, maxval=1, device=None):
        super().__init__()
        self.shape = shape
        self.device = self.device = device if device else T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.minval = T.tensor(minval, device=self.device)
        self.maxval = T.tensor(maxval, device=self.device)
        
        self.noise_gen = uniform.Uniform(low=self.minval, high=self.maxval)

    def __call__(self):
        return self.noise_gen.sample(self.shape)

    def get_config(self):
        return {
            'class_name': 'UniformNoise',
            'config': {
                'shape': self.shape,
                'minval': self.minval.item(),
                'maxval': self.maxval.item(),
                'device': self.device,
            }
        }
    
    def clone(self):
        return UniformNoise(
            self.shape,
            self.minval,
            self.maxval,
            self.device
        )

class NormalNoise:
    def __init__(self, shape, mean=0.0, stddev=1.0, device=None):
        super().__init__()
        self.shape = shape
        self.device = device if device else T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.mean = np.array(mean, dtype=np.float32)
        self.stddev = np.array(stddev, dtype=np.float32)

        # Initialize the noise generator here using the numpy arrays
        self.reset_noise_gen()

    def reset_noise_gen(self):
        # Convert numpy mean and stddev to tensors just for noise generation
        mean_tensor = T.tensor(self.mean, device=self.device)
        stddev_tensor = T.tensor(self.stddev, device=self.device)
        self.noise_gen = normal.Normal(loc=mean_tensor, scale=stddev_tensor)

    def __call__(self):
        # Directly sample using the noise generator
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

    def get_config(self):
        return {
            'class_name': 'NormalNoise',
            'config': {
                'shape': self.shape,
                'mean': self.mean.item(),
                'stddev': self.stddev.item(),
                'device': self.device,
            }
        }
    
    def clone(self):
        return NormalNoise(
            self.shape,
            self.mean,
            self.stddev,
            self.device
        )
    
class OUNoise(Noise):
    """Ornstein-Uhlenbeck noise process."""

    def __init__(self, shape: tuple, mean: float = 0.0, theta: float = 0.15, sigma: float = 0.2, dt: float = 1e-2, device=None):
        """Initializes a new Ornstein-Uhlenbeck noise process."""
        super().__init__()
        self.device = device if device else T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.shape = shape
        self.mean = T.tensor(mean, device=self.device)
        self.mu = T.ones(self.shape, device=self.device) * self.mean
        self.theta = T.tensor(theta, device=self.device)
        self.sigma = T.tensor(sigma, device=self.device)
        self.dt = T.tensor(dt, device=self.device)
        self.x_prev = T.ones(self.shape, device=self.device) * self.mean

    def __call__(self):
        """Samples a new noise vector."""
        dx = self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * T.randn(self.shape, device=self.device)
        x = self.x_prev + dx
        self.x_prev = x
        return x

    def reset(self, mu: T.tensor = None):
        """Resets the noise process."""
        self.mu = T.ones(self.shape, device=self.device) * self.mean if mu is None else T.tensor(mu, device=self.device)
        self.x_prev = T.ones(self.shape, device=self.device) * self.mu

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                "shape": self.shape,
                "mean": self.mean.item(),
                "theta": self.theta.item(),
                "sigma": self.sigma.item(),
                "dt": self.dt.item(),
                'device': self.device,
            }
        }
        
    def clone(self):
        return OUNoise(
            self.shape,
            self.mean,
            self.theta,
            self.sigma,
            self.dt,
            self.device
        )