import torch as T
import numpy as np
from typing import Optional
from .torch_utils import get_device
from .logging_config import get_logger


def create_normalizer(config: dict) -> 'BaseNormalizer':
    normalizer_type = config['type']
    if normalizer_type not in NORMALIZER_CLASSES:
        raise ValueError(f"Invalid normalizer type: {normalizer_type}")
    return NORMALIZER_CLASSES[normalizer_type](**config['config'])

class BaseNormalizer:
    """
    Base Normalizer class.

    Attributes:
        num_features (int): Number of features to normalize.
        clip_value (float): Value to clip normalized values.
        epsilon (float): Small constant to prevent division by zero.
        device (str): Device to run the normalizer on ('cpu' or 'cuda').
        log_level (str): Log level for the normalizer.
        **kwargs: Additional keyword arguments.
    """
    def __init__(
        self,
        num_features: int,
        clip_value: float = 5.0,
        epsilon: float = 1e-6,
        device: Optional[str | T.device] = None,
        log_level: str = 'INFO',
        **kwargs
    ):
        self.logger = get_logger(self.__class__.__name__, level=log_level.upper())
        self.kwargs = kwargs
        self.device = get_device(device)
        self.num_features = num_features
        self.clip_value = T.tensor(clip_value, device=self.device)
        self.epsilon = T.tensor(epsilon, device=self.device)
        # Local statistics
        self.local_cnt = T.zeros(1, dtype=T.int32, device=self.device)
        self.local_sum = T.zeros(self.num_features, dtype=T.float32, device=self.device)
        self.local_sum_sq = T.zeros(self.num_features, dtype=T.float32, device=self.device)
        # Running statistics
        self.running_cnt = T.zeros(1, dtype=T.int32, device=self.device)
        self.running_sum = T.zeros(self.num_features, dtype=T.float32, device=self.device)
        self.running_sum_sq = T.zeros(self.num_features, dtype=T.float32, device=self.device)
        self.running_mean = T.zeros(self.num_features, dtype=T.float32, device=self.device)
        self.running_var = T.ones(self.num_features, dtype=T.float32, device=self.device)
        self.running_std = T.ones(self.num_features, dtype=T.float32, device=self.device)

        # Set training bool to True
        self.training = True

        # Set internal attributes
        self.step = 0
        self._diag_freq = None
        self._log_diag = False
        if self.kwargs is not None:
            for key, value in self.kwargs.items():
                setattr(self, key, value)

    def add(self, new_data: T.Tensor) -> None:
        """
        Update local statistics with new data.

        Args:
            new_data (T.Tensor): New data to update local statistics.
        """
        self.local_sum += new_data.sum(dim=0).to(self.device)
        self.local_sum_sq += (new_data**2).sum(dim=0).to(self.device)
        self.local_cnt += new_data.size(0)

        # Log diag values if diag
        if self._diag_freq is not None:
            should_log_diag = (self.step % self._diag_freq == 0)
        else:
            should_log_diag = False
        if should_log_diag:
            self.logger.debug(f"Normalizer add: step={self.step}, data={new_data}, data_shape={new_data.shape}, local_sum={self.local_sum}, local_sum_sq={self.local_sum_sq}, local_cnt={self.local_cnt}, running_cnt={self.running_cnt}, running_sum={self.running_sum}, running_sum_sq={self.running_sum_sq}")

    def update(self) -> None:
        """
        Update running statistics based on local statistics.
        """
        self.step += 1
        # self.running_cnt += self.local_cnt
        # self.running_sum += self.local_sum
        # self.running_sum_sq += self.local_sum_sq
        # self.running_mean = self.running_sum / self.running_cnt
        # self.running_var = self.running_sum_sq / self.running_cnt - self.running_mean.pow(2)
        if self.local_cnt.item() == 0:
            return

        batch_cnt = self.local_cnt.item()
        batch_mean = self.local_sum / self.local_cnt
        batch_var = self.local_sum_sq / self.local_cnt - batch_mean**2

        if self.running_cnt.item() == 0:
            self.running_mean.copy_(batch_mean)
            self.running_var.copy_(batch_var)
            self.running_cnt.copy_(self.local_cnt)
        else:
            total_cnt = self.running_cnt + batch_cnt
            delta = batch_mean - self.running_mean

            self.running_mean.add_(delta * (batch_cnt / total_cnt))

            m_a = self.running_var * self.running_cnt
            m_b = batch_var * batch_cnt
            m2 = m_a + m_b + delta**2 * (self.running_cnt * batch_cnt / total_cnt)
            self.running_var.copy_(m2 / total_cnt)

            self.running_cnt.add_(batch_cnt)
            self.running_std = T.sqrt(self.running_var + self.epsilon**2).clamp(min=1e-4)
        # Log diag values if diag
        if self._diag_freq is not None:
            self._log_diag = (self.step % self._diag_freq == 0)
        if self._log_diag:
            self.logger.debug(f"Normalizer update: step={self.step}, running_cnt={self.running_cnt}, running_sum={self.running_sum}, running_sum_sq={self.running_sum_sq}, running_mean={self.running_mean}, running_var={self.running_var}, running_std={self.running_std}")

        # Reset local statistics
        self.local_cnt.zero_()
        self.local_sum.zero_()
        self.local_sum_sq.zero_()

    def denormalize(self, v: T.Tensor) -> T.Tensor:
        """
        Denormalize a tensor using running statistics.

        Args:
            v (T.Tensor): Input tensor to denormalize.

        Returns:
            T.Tensor: Denormalized tensor.
        """
        if v.device != self.device:
            v = v.to(self.device)
        return (v * self.running_std) + self.running_mean

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def get_config(self) -> dict:
        """
        Retrieve the configuration and state of the normalizer.

        Returns:
            dict: Configuration and state of the normalizer.
        """
        return {
            'type': self.__class__.__name__,
            'config': {
                'num_features':self.num_features,
                'epsilon':self.epsilon.item(),
                'clip_value':self.clip_value.item(),
                'device':self.device.type,
            },
        }

    def save(self, file_path: str) -> None:
        """
        Save the current state of the normalizer to a file.

        Args:
            file_path (str): Path to save the state.
        """
        T.save({
            'step': self.step,
            'local_sum': self.local_sum.cpu().detach().numpy(),
            'local_sum_sq': self.local_sum_sq.cpu().detach().numpy(),
            'local_cnt': self.local_cnt.cpu().detach().numpy(),
            'running_cnt': self.running_cnt.cpu().detach().numpy(),
            'running_sum': self.running_sum.cpu().detach().numpy(),
            'running_sum_sq': self.running_sum_sq.cpu().detach().numpy(),
            'running_mean': self.running_mean.cpu().detach().numpy(),
            'running_var': self.running_var.cpu().detach().numpy(),
            'running_std': self.running_std.cpu().detach().numpy(),
        }, file_path)

    @classmethod
    def load(cls, config: dict, state_path: str) -> 'BaseNormalizer':
        """
        Load a BaseNormalizer state from a file.

        Args:
            config (dict): Configuration of the normalizer.
            state_path (str): Path to load the state from.

        Returns:
            BaseNormalizer: A BaseNormalizer instance with the loaded state.
        """

        norm_type = config['type']
        if norm_type not in NORMALIZER_CLASSES:
            raise ValueError(f"Invalid normalizer type: {norm_type}")
        return NORMALIZER_CLASSES[norm_type].load(config, state_path)

class RunningNorm(BaseNormalizer):
    """
    Normalizes data using running statistics (mean and standard deviation).

    Attributes:
        num_features (int): Number of features to normalize.
        clip_value (float): Value to clip normalized values.
        epsilon (float): Small constant to prevent division by zero.
        warmup_steps (int): Number of warmup steps.
        device (str): Device to run the normalizer on ('cpu' or 'cuda').
    """
    def __init__(
        self,
        num_features: int,
        clip_value: float = 5.0,
        epsilon: float = 1e-6,
        warmup_steps: int = 1000,
        device: Optional[str | T.device] = None,
        log_level: str = 'INFO',
        **kwargs
    ):
        super().__init__(num_features, clip_value, epsilon, device, log_level, **kwargs)
        self.warmup_steps = warmup_steps

    def normalize(self, v: T.Tensor) -> T.Tensor:
        """
        Normalize a tensor using running statistics.

        Args:
            v (T.Tensor): Input tensor to normalize.

        Returns:
            T.Tensor: Normalized tensor.
        """
        if v.device != self.device:
            v = v.to(self.device)

        if self.training and self.step <= self.warmup_steps:
            return v

        norms = T.clamp((v - self.running_mean) / self.running_std,
                       -self.clip_value, self.clip_value).float()
        # Log diag values if diag
        if self._log_diag:
            self.logger.debug(f"RunningNorm normalize: step={self.step}, data={v}, data_shape={v.shape}, running_mean={self.running_mean}, running_std={self.running_std}, norms={norms}")
        return norms

    def get_config(self) -> dict:
        config = super().get_config()
        config['type'] = self.__class__.__name__
        config['config'].update({
            'warmup_steps': self.warmup_steps,
        })
        return config

    @classmethod
    def load(cls, config: dict, state_path: str) -> 'RunningNorm':
        """
        Load a RunningNorm state from a file.

        Args:
            config (dict): Configuration of the normalizer.
            state_path (str): Path to load the state from.

        Returns:
            RunningNorm: A RunningNorm instance with the loaded state.
        """

        device = get_device(config['device'])
        state = T.load(state_path, map_location='cpu', weights_only=False)
        normalizer = RunningNorm(
            num_features=config['num_features'],
            clip_value=config['clip_value'],
            epsilon=config['epsilon'],
            warmup_steps=config['warmup_steps'],
            device=config['device']
        )
        normalizer.step = state['step']
        normalizer.local_sum = T.tensor(state['local_sum'], device=device)
        normalizer.local_sum_sq = T.tensor(state['local_sum_sq'], device=device)
        normalizer.local_cnt = T.tensor(state['local_cnt'], device=device)
        normalizer.running_cnt = T.tensor(state['running_cnt'], device=device)
        normalizer.running_sum = T.tensor(state['running_sum'], device=device)
        normalizer.running_sum_sq = T.tensor(state['running_sum_sq'], device=device)
        normalizer.running_mean = T.tensor(state['running_mean'], device=device)
        normalizer.running_var = T.tensor(state['running_var'], device=device)
        normalizer.running_std = T.tensor(state['running_std'], device=device)

        return normalizer

class BatchNorm(BaseNormalizer):
    """
    Normalizes data using batch statistics (mean and standard deviation).

    Attributes:
        num_features (int): Number of features to normalize.
        clip_value (float): Value to clip normalized values.
        epsilon (float): Small constant to prevent division by zero.
        device (str): Device to run the normalizer on ('cpu' or 'cuda').
    """
    def __init__(
        self,
        num_features: int,
        clip_value: float = 5.0,
        epsilon: float = 1e-6,
        device: Optional[str | T.device] = None,
        log_level: str = 'INFO',
        **kwargs
    ):
        super().__init__(num_features, clip_value, epsilon, device, log_level, **kwargs)

    def normalize(self, v: T.Tensor) -> T.Tensor:
        """
        Normalize a tensor using batch statistics during training, running statistics during evaluation.
        """
        if v.device != self.device:
            v = v.to(self.device)

        if self.training:
            mean = v.mean(dim=0, keepdim=True)
            var = v.var(dim=0, unbiased=False, keepdim=True)
            std = T.sqrt(var + self.epsilon**2).clamp(min=1e-4)
            norms = (v - mean) / std
        else:
            norms = (v - self.running_mean) / self.running_std

        if self._log_diag:
            self.logger.debug(f"BatchNorm normalize: step={self.step}, data={v}, data_shape={v.shape}, running_mean={self.running_mean}, running_std={self.running_std}, norms={norms}")

        return T.clamp(norms, -self.clip_value, self.clip_value).float()

    def get_config(self) -> dict:
        config = super().get_config()
        config['type'] = self.__class__.__name__
        return config

    @classmethod
    def load(cls, config: dict, state_path: str) -> 'BatchNorm':
        """
        Load a BatchNorm state from a file.

        Args:
            config (dict): Configuration of the normalizer.
            state_path (str): Path to load the state from.

        Returns:
            BatchNorm: A BatchNorm instance with the loaded state.
        """

        device = get_device(config['device'])
        state = T.load(state_path, map_location='cpu', weights_only=False)
        normalizer = BatchNorm(
            num_features=config['num_features'],
            clip_value=config['clip_value'],
            epsilon=config['epsilon'],
            device=config['device']
        )
        normalizer.step = state['step']
        normalizer.local_sum = T.tensor(state['local_sum'], device=device)
        normalizer.local_sum_sq = T.tensor(state['local_sum_sq'], device=device)
        normalizer.local_cnt = T.tensor(state['local_cnt'], device=device)
        normalizer.running_cnt = T.tensor(state['running_cnt'], device=device)
        normalizer.running_sum = T.tensor(state['running_sum'], device=device)
        normalizer.running_sum_sq = T.tensor(state['running_sum_sq'], device=device)
        normalizer.running_mean = T.tensor(state['running_mean'], device=device)
        normalizer.running_var = T.tensor(state['running_var'], device=device)
        normalizer.running_std = T.tensor(state['running_std'], device=device)

        return normalizer

class RewardNormalizer(BaseNormalizer):
    """
    Normalizes rewards using running return standard deviation.
    """
    def __init__(
        self,
        gamma: float = 0.99,
        clip_value: float = 5.0,
        epsilon: float = 1e-6,
        device: Optional[str | T.device] = None,
        log_level: str = 'INFO',
        **kwargs
    ):
        super().__init__(1, clip_value, epsilon, device, log_level, **kwargs)
        
class SharedNormalizer:
    def __init__(self, size, eps=1e-2, clip_range=5.0):
        self.size = size
        self.eps = eps
        self.clip_range = clip_range

        # self.lock = manager.Lock()
        self.lock = threading.Lock()

        # Create shared memory blocks
        total_byte_size = np.prod(self.size) * np.float32().itemsize
        self.shared_local_sum = shared_memory.SharedMemory(create=True, size=total_byte_size)
        self.shared_local_sum_sq = shared_memory.SharedMemory(create=True, size=total_byte_size)
        self.shared_local_cnt = shared_memory.SharedMemory(create=True, size=np.float32().itemsize)

        self.local_sum = np.ndarray(self.size, dtype=np.float32, buffer=self.shared_local_sum.buf)
        self.local_sum_sq = np.ndarray(self.size, dtype=np.float32, buffer=self.shared_local_sum_sq.buf)
        self.local_cnt = np.ndarray(1, dtype=np.int32, buffer=self.shared_local_cnt.buf)

        # Initiate shared arrays to zero
        self.local_sum.fill(0)
        self.local_sum_sq.fill(0)
        self.local_cnt.fill(0)

        self.running_mean = np.zeros(self.size, dtype=np.float32)
        self.running_std = np.ones(self.size, dtype=np.float32)
        self.running_sum = np.zeros(self.size, dtype=np.float32)
        self.running_sum_sq = np.zeros(self.size, dtype=np.float32)
        self.running_cnt = np.zeros(1, dtype=np.int32)

    def normalize(self, v):
        clip_range = self.clip_range
        return np.clip((v - self.running_mean) / self.running_std,
                       -clip_range, clip_range).astype(np.float32)
    
    def update_local_stats(self, new_data):
        # print('SharedNormalizer update_local_stats fired...')
        try:
            with self.lock:
                # print('SharedNormalizer update_local_stats lock acquired...')
                # print(f'data: {new_data}')
                # print('previous local stats')
                # print(f'local sum: {self.local_sum}')
                # print(f'local sum sq: {self.local_sum_sq}')
                # print(f'local_cnt: {self.local_cnt}')
                self.local_sum += new_data#.sum(axis=1)
                self.local_sum_sq += (np.square(new_data))#.sum(axis=1)
                self.local_cnt += 1 #new_data.shape[0]
                # print('new local values')
                # print(f'local sum: {self.local_sum}')
                # print(f'local sum sq: {self.local_sum_sq}')
                # print(f'local_cnt: {self.local_cnt}')
        except Exception as e:
            print(f"Error during update: {e}")
    
    def update_global_stats(self):
        with self.lock:
            # make copies of local stats
            local_cnt = self.local_cnt.copy()
            local_sum = self.local_sum.copy()
            local_sum_sq = self.local_sum_sq.copy()
            
            # Zero out local stats
            self.local_cnt[...] = 0
            self.local_sum[...] = 0
            self.local_sum_sq[...] = 0
            
            # Add local stats to global stats
            self.running_cnt += local_cnt
            self.running_sum += local_sum
            self.running_sum_sq += local_sum_sq

            # Calculate new mean, sum_sq, and std
            self.running_mean = self.running_sum / self.running_cnt
            tmp = self.running_sum_sq / self.running_cnt -\
                np.square(self.running_sum / self.running_cnt)
            self.running_std = np.sqrt(np.maximum(np.square(self.eps), tmp))

    def get_config(self):
        return {
            "params":{
                'size':self.size,
                'eps':self.eps,
                'clip_range':self.clip_range,
            },
            "state":{
                'local_sum':self.local_sum,
                'local_sum_sq':self.local_sum_sq,
                'local_cnt':self.local_cnt,
                'running_mean':self.running_mean,
                'running_std':self.running_std,
                'running_sum':self.running_sum,
                'running_sum_sq':self.running_sum_sq,
                'running_cnt':self.running_cnt,
            },
        }


    def save_state(self, file_path):
        np.savez(
            file_path,
            local_sum=self.local_sum,
            local_sum_sq=self.local_sum_sq,
            local_cnt=self.local_cnt,
            running_mean=self.running_mean,
            running_std=self.running_std,
            running_sum=self.running_sum,
            running_sum_sq=self.running_sum_sq,
            running_cnt=self.running_cnt,
        )

    def cleanup(self):
        # Close and unlink shared memory blocks
        try:
            if self.shared_local_sum:
                self.shared_local_sum.unlink()
                self.shared_local_sum.close()
                self.shared_local_sum = None
        except FileNotFoundError as e:
            print(f"Shared local sum already cleaned up: {e}")
        try:
            if self.shared_local_sum_sq:
                self.shared_local_sum_sq.unlink()
                self.shared_local_sum_sq.close()
                self.shared_local_sum_sq = None
        except FileNotFoundError as e:
            print(f"Shared local sum sq already cleaned up: {e}")
        try:
            if self.shared_local_cnt:
                self.shared_local_cnt.unlink()
                self.shared_local_cnt.close()
                self.shared_local_cnt = None
        except FileNotFoundError as e:
            print(f"Shared local sum cnt already cleaned up: {e}")

        print("SharedNormalizer resources have been cleaned up.")

    def __del__(self):
        self.cleanup()


    @classmethod
    def load_state(cls, file_path):
        with np.load(file_path) as data:
            normalizer = cls(size=data['running_mean'].shape)
            normalizer.local_sum = data['local_sum']
            normalizer.local_sum_sq = data['local_sum_sq']
            normalizer.local_cnt = data['local_cnt']
            normalizer.running_mean = data['running_mean']
            normalizer.running_std = data['running_std']
            normalizer.running_sum = data['running_sum']
            normalizer.running_sum_sq = data['running_sum_sq']
            normalizer.running_cnt = data['running_cnt']
        return normalizer
    
    @classmethod
    def create_instance(cls, **kwargs) -> 'SharedNormalizer':
        return cls(**kwargs)

NORMALIZER_CLASSES = {
    "RunningNorm": RunningNorm,
    "BatchNorm": BatchNorm,
}