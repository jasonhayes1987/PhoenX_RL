import torch as T
import numpy as np
from torch_utils import get_device
from typing import Optional

class Normalizer:
    """
    Normalizes data using running statistics (mean and standard deviation).

    Attributes:
        size (int): Size of the input data to normalize.
        eps (float): Small constant to prevent division by zero.
        clip_range (float): Range to clip normalized values.
        device (str): Device to run the normalizer on ('cpu' or 'cuda').
    """
    def __init__(self, size: int, eps: float = 1e-2, clip_range: float = 5.0, device: Optional[str | T.device] = None):
        self.size = size
        self.device = get_device(device)
        self.eps = T.tensor(eps, device=self.device)
        self.clip_range = T.tensor(clip_range, device=self.device)

        # Local statistics
        self.local_sum = T.zeros(self.size, dtype=T.float32, device=self.device)
        self.local_sum_sq = T.zeros(self.size, dtype=T.float32, device=self.device)
        self.local_cnt = T.zeros(1, dtype=T.int32, device=self.device)

        # Running statistics
        self.running_mean = T.zeros(self.size, dtype=T.float32, device=self.device)
        self.running_std = T.ones(self.size, dtype=T.float32, device=self.device)
        self.running_sum = T.zeros(self.size, dtype=T.float32, device=self.device)
        self.running_sum_sq = T.zeros(self.size, dtype=T.float32, device=self.device)
        self.running_cnt = T.zeros(1, dtype=T.int32, device=self.device)

    def normalize(self, v: T.Tensor) -> T.Tensor:
        """
        Normalize a tensor using running statistics.

        Args:
            v (T.Tensor): Input tensor to normalize.

        Returns:
            T.Tensor: Normalized tensor.
        """
        # Ensure input tensor is on the same device as normalizer
        if v.device != self.device:
            v = v.to(self.device)
        
        return T.clamp((v - self.running_mean) / self.running_std,
                       -self.clip_range, self.clip_range).float()

    def denormalize(self, v: T.Tensor) -> T.Tensor:
        """
        Denormalize a tensor using running statistics.

        Args:
            v (T.Tensor): Input tensor to denormalize.

        Returns:
            T.Tensor: Denormalized tensor.
        """
        return (v * self.running_std) + self.running_mean
    
    def update_local_stats(self, new_data: T.Tensor) -> None:
        """
        Update local statistics with new data.

        Args:
            new_data (T.Tensor): New data to update local statistics.
        """
        try:
            self.local_sum += new_data.sum(dim=0).to(self.device)
            self.local_sum_sq += (new_data**2).sum(dim=0).to(self.device)
            self.local_cnt += new_data.size(0)
        except Exception as e:
            print(f"Error during local stats update: {e}")
    
    def update_global_stats(self) -> None:
        """
        Update running statistics based on local statistics.
        """

        self.running_cnt += self.local_cnt
        self.running_sum += self.local_sum
        self.running_sum_sq += self.local_sum_sq

        self.local_cnt.zero_()
        self.local_sum.zero_()
        self.local_sum_sq.zero_()

        # Ensure all calculations remain on the correct device
        self.running_mean = (self.running_sum / self.running_cnt).to(self.device)
        tmp = (self.running_sum_sq / self.running_cnt - (self.running_sum / self.running_cnt)**2).to(self.device)
        eps_squared = self.eps**2
        self.running_std = T.sqrt(T.maximum(eps_squared, tmp)).to(self.device)

    def get_config(self) -> dict:
        """
        Retrieve the configuration and state of the normalizer.

        Returns:
            dict: Configuration and state of the normalizer.
        """
        return {
            "params":{
                'size':self.size,
                'eps':self.eps,
                'clip_range':self.clip_range,
                'device':self.device.type,
            },
            "state":{
                'local_sum':self.local_sum.cpu().numpy(),
                'local_sum_sq':self.local_sum_sq.cpu().numpy(),
                'local_cnt':self.local_cnt.cpu().numpy(),
                'running_mean':self.running_mean.cpu().numpy(),
                'running_std':self.running_std.cpu().numpy(),
                'running_sum':self.running_sum.cpu().numpy(),
                'running_sum_sq':self.running_sum_sq.cpu().numpy(),
                'running_cnt':self.running_cnt.cpu().numpy(),
            },
        }

    def save_state(self, file_path: str) -> None:
        """
        Save the current state of the normalizer to a file.

        Args:
            file_path (str): Path to save the state.
        """
        T.save({
            'local_sum': self.local_sum.cpu().numpy(),
            'local_sum_sq': self.local_sum_sq.cpu().numpy(),
            'local_cnt': self.local_cnt.cpu().numpy(),
            'running_mean': self.running_mean.cpu().numpy(),
            'running_std': self.running_std.cpu().numpy(),
            'running_sum': self.running_sum.cpu().numpy(),
            'running_sum_sq': self.running_sum_sq.cpu().numpy(),
            'running_cnt': self.running_cnt.cpu().numpy(),
        }, file_path)

    @classmethod
    def load_state(cls, file_path: str, device: Optional[str] = None) -> 'Normalizer':
        """
        Load a normalizer's state from a file.

        Args:
            file_path (str): Path to load the state from.
            device (str): Device to load the state to ('cpu' or 'cuda').

        Returns:
            Normalizer: A Normalizer instance with the loaded state.
        """
        state = T.load(file_path)
        device = get_device(device)
        normalizer = cls(size=state['running_mean'].shape[0], device=device)
        target_device = normalizer.device
        
        # Ensure all loaded tensors are moved to the correct device
        normalizer.local_sum = T.tensor(state['local_sum'], device=target_device)
        normalizer.local_sum_sq = T.tensor(state['local_sum_sq'], device=target_device)
        normalizer.local_cnt = T.tensor(state['local_cnt'], device=target_device)
        normalizer.running_mean = T.tensor(state['running_mean'], device=target_device)
        normalizer.running_std = T.tensor(state['running_std'], device=target_device)
        normalizer.running_sum = T.tensor(state['running_sum'], device=target_device)
        normalizer.running_sum_sq = T.tensor(state['running_sum_sq'], device=target_device)
        normalizer.running_cnt = T.tensor(state['running_cnt'], device=target_device)
        
        return normalizer

    
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