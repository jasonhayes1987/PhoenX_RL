import numpy as np
import torch as T
from torch.distributions import Distribution, TransformedDistribution, TanhTransform, AffineTransform, Beta, Kumaraswamy, Normal
from torch.distributions.utils import _sum_rightmost

class TanhBijector:
    """
    Bijective tanh transformation with numerical safeguards for stable squashing 
    of unbounded Gaussians to bounded continuous action spaces.
    Provides forward (tanh), inverse (clamped atanh), and Jacobian correction.
    """
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def atanh(self, x: T.Tensor) -> T.Tensor:
        """Stable atanh."""
        x = x.clamp(min=-1.0 + self.epsilon, max=1.0 - self.epsilon)
        return 0.5 * (x.log1p() - (-x).log1p())

    # @staticmethod
    # def inverse(y: T.Tensor) -> T.Tensor:
    #     """Inverse tanh with clamping."""
    #     # eps = T.finfo(y.dtype).eps
    #     return TanhBijector.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    def log_prob_correction(self, x: T.Tensor) -> T.Tensor:
        """log|det J_tanh| = log(1 - tanh²(x))"""
        return T.log(1.0 - T.tanh(x) ** 2 + self.epsilon)


class SquashedNormal(Distribution):
    """
    Squashed Normal distribution that inherits from torch.distributions.Distribution.
    - Samples directly in [low, high].
    - Correct log_prob with full Jacobian correction.
    - Works with Independent(...), KL, etc.
    - No NaNs.
    """
    def __init__(
        self,
        base_dist: Normal,
        low: T.Tensor | np.ndarray | float,
        high: T.Tensor | np.ndarray | float,
        epsilon: float = 1e-5,
        validate_args: bool = False,
    ):
        self.base_dist = base_dist
        self.epsilon = epsilon

        self.low = T.as_tensor(low, dtype=T.float32, device=base_dist.loc.device).flatten()
        self.high = T.as_tensor(high, dtype=T.float32, device=base_dist.loc.device).flatten()

        self.loc = (self.low + self.high) / 2.0
        self.scale = (self.high - self.low) / 2.0
        self.bijector = TanhBijector(epsilon)

        super().__init__(
            batch_shape=self.base_dist.batch_shape,
            event_shape=self.base_dist.event_shape,
            validate_args=validate_args,
        )

    def rsample(self, sample_shape=T.Size()):
        z = self.base_dist.rsample(sample_shape)
        y = T.tanh(z)
        return self.loc + self.scale * y

    def sample(self, sample_shape=T.Size()):
        with T.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value: T.Tensor) -> T.Tensor:
        """Returns (B, D) — Independent wrapper will sum to (B,)"""
        value = value.clamp(self.low + self.epsilon, self.high - self.epsilon)
        y = (value - self.loc) / self.scale
        y = y.clamp(-1.0 + self.epsilon, 1.0 - self.epsilon)
        z = self.bijector.atanh(y)

        log_prob = self.base_dist.log_prob(z)
        # log_prob -= self.bijector.log_prob_correction(z)
        # log_prob -= T.log(self.scale)
        log_prob = log_prob - T.log(self.scale) - T.log(1.0 - y.pow(2) + self.epsilon)

        return log_prob

    def entropy(self) -> T.Tensor:
        """Returns (B, D) per-dimension entropy — Independent will handle reduction"""
        mc_samples = 8
        z = self.base_dist.rsample((mc_samples,))

        y = T.tanh(z)
        log_det_tanh = T.log(1.0 - y.pow(2) + self.epsilon)

        base_entropy = self.base_dist.entropy()
        affine_term = T.log(self.scale)

        return base_entropy + log_det_tanh.mean(0) + affine_term

# class SquashedNormal(TransformedDistribution):
#     """
#     Squashed Normal distribution for continuous actions.

#     Args:
#         base_dist (Normal): The base normal distribution.
#         low (float): The lower bound of the action space.
#         high (float): The upper bound of the action space.
#     """
#     def __init__(
#         self,
#         base_dist: Normal,
#         low:T.Tensor|np.ndarray,
#         high:T.Tensor|np.ndarray,
#         mc_samples:int = 8,
#         epsilon:float = 1e-6
#     ):
#         self.low = T.as_tensor(low, dtype=T.float32, device=base_dist.loc.device)
#         self.high = T.as_tensor(high, dtype=T.float32, device=base_dist.loc.device)
#         self.mc_samples = mc_samples
#         self.epsilon = epsilon
#         self.scale = T.as_tensor((high - low) / 2.0, dtype=T.float32, device=base_dist.loc.device)
#         # self.log_scale = T.log(self.scale).sum()
#         self.loc = T.as_tensor((high + low) / 2.0, dtype=T.float32, device=base_dist.loc.device)
#         transforms = [
#             TanhTransform(cache_size=1),
#             AffineTransform(loc=self.loc, scale=self.scale, cache_size=1),
#         ]
#         super().__init__(base_dist, transforms)

#     def log_prob(self, values: T.Tensor) -> T.Tensor:
#         values = values.clamp(self.low + self.epsilon, self.high - self.epsilon)
#         return super().log_prob(values)

#     def entropy(self) -> T.Tensor:
#         # Sample from the *base* Normal (pre-tanh)
#         z = self.base_dist.rsample((self.mc_samples,))          # (mc_samples, batch, action_dim)

#         # Tanh transform (first transform)
#         tanh_z = T.tanh(z)                                      # u = tanh(z)

#         # Jacobian correction: log|det J_tanh| = sum log(1 - tanh²(z_i))
#         log_det_jacobian = T.log(1 - tanh_z.pow(2) + self.epsilon)#.sum(-1)   # negative value

#         # Base entropy (already summed over action dims by Normal)
#         base_entropy = self.base_dist.entropy()        # (batch,)

#         # Affine scale term (constant)
#         log_scale = T.log(self.scale)#.sum()                     # scalar (broadcasts)

#         # Monte-Carlo average
#         entropy = base_entropy + log_det_jacobian.mean(0) + log_scale

#         # Safety clamp (entropy cannot be negative)
#         # entropy = T.clamp(entropy, min=self.epsilon)

#         return entropy

#     # def entropy(self) -> T.Tensor:
#     #     return self.base_dist.entropy().sum(-1)

class ScaledBeta(TransformedDistribution):
    """
    Scaled Beta distribution to low/high bounds.

    Args:
        base_dist (Beta): The base beta distribution.
        low (float): The lower bound of the action space.
        high (float): The upper bound of the action space.
    """
    def __init__(
        self,
        base_dist: Beta,
        low:T.Tensor|np.ndarray,
        high:T.Tensor|np.ndarray
    ):
        self.low = T.as_tensor(low, dtype=T.float32, device=base_dist.concentration0.device)
        self.high = T.as_tensor(high, dtype=T.float32, device=base_dist.concentration0.device)
        scale = T.as_tensor((high - low), dtype=T.float32, device=base_dist.concentration0.device)
        transforms = [AffineTransform(loc=self.low, scale=scale, cache_size=1)]
        super().__init__(base_dist, transforms)
        self.log_scale = T.log(scale)

    def entropy(self) -> T.Tensor:
        return self.base_dist.entropy() + self.log_scale

class ScaledKumaraswamy(TransformedDistribution):
    """
    Scaled Kumaraswamy distribution to low/high bounds.

    Args:
        base_dist (Kumaraswamy): The base kumaraswamy distribution.
        low (float): The lower bound of the action space.
        high (float): The upper bound of the action space.
    """
    def __init__(self, base_dist: Kumaraswamy, low:float = 0.0, high:float = 1.0):
        scale = T.tensor(high - low, device=base_dist.concentration0.device)
        transforms = [AffineTransform(loc=low, scale=scale, cache_size=1)]
        super().__init__(base_dist, transforms)
        self.log_scale = T.log(scale)

    def entropy(self) -> T.Tensor:
        return self.base_dist.entropy() + self.log_scale