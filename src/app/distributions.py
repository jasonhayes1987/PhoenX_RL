import numpy as np
import torch as T
from torch.distributions import TransformedDistribution, TanhTransform, AffineTransform, Beta, Kumaraswamy, Normal

class SquashedNormal(TransformedDistribution):
    """
    Squashed Normal distribution for continuous actions.

    Args:
        base_dist (Normal): The base normal distribution.
        low (float): The lower bound of the action space.
        high (float): The upper bound of the action space.
    """
    def __init__(
        self,
        base_dist: Normal,
        low:T.Tensor|np.ndarray,
        high:T.Tensor|np.ndarray,
        mc_samples:int = 8,
        epsilon:float = 1e-6
    ):
        self.low = T.as_tensor(low, dtype=T.float32, device=base_dist.loc.device)
        self.high = T.as_tensor(high, dtype=T.float32, device=base_dist.loc.device)
        # self.mc_samples = mc_samples
        self.epsilon = epsilon
        self.scale = T.as_tensor((high - low) / 2.0, dtype=T.float32, device=base_dist.loc.device)
        # self.log_scale = T.log(self.scale.abs()).sum()
        self.loc = T.as_tensor((high + low) / 2.0, dtype=T.float32, device=base_dist.loc.device)
        transforms = [
            TanhTransform(cache_size=1),
            AffineTransform(loc=self.loc, scale=self.scale, cache_size=1),
        ]
        super().__init__(base_dist, transforms)

    def log_prob(self, values: T.Tensor) -> T.Tensor:
        values = values.clamp(self.low + self.epsilon, self.high - self.epsilon)
        return super().log_prob(values)

    # def entropy(self) -> T.Tensor:
    #     z = self.base_dist.rsample((self.mc_samples,))
    #     tanh_transform = self.transforms[0]
    #     u = tanh_transform(z)
    #     base_entropy = self.base_dist.entropy().sum(-1)
    #     log_det_tanh = tanh_transform.log_abs_det_jacobian(z, u).sum(-1)
    #     return base_entropy + log_det_tanh.mean(0) + self.log_scale

    def entropy(self) -> T.Tensor:
        return self.base_dist.entropy().sum(-1)

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
        self.log_scale = T.log(scale).sum()

    def entropy(self) -> T.Tensor:
        return self.base_dist.entropy().sum(-1) + self.log_scale

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
        self.log_scale = T.log(scale).sum()

    def entropy(self) -> T.Tensor:
        return self.base_dist.entropy().sum(-1) + self.log_scale