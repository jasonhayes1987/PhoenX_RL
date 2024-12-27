import torch as T


class Normalizer:
    def __init__(self, size, eps=1e-2, clip_range=5.0, device='cpu'):
        self.size = size
        self.eps = T.tensor(eps, device=device)
        # self.eps = eps
        self.clip_range = clip_range
        self.device = device

        self.local_sum = T.zeros(self.size, dtype=T.float32, device=self.device)
        self.local_sum_sq = T.zeros(self.size, dtype=T.float32, device=self.device)
        self.local_cnt = T.zeros(1, dtype=T.int32, device=self.device)

        self.running_mean = T.zeros(self.size, dtype=T.float32, device=self.device)
        self.running_std = T.ones(self.size, dtype=T.float32, device=self.device)
        self.running_sum = T.zeros(self.size, dtype=T.float32, device=self.device)
        self.running_sum_sq = T.zeros(self.size, dtype=T.float32, device=self.device)
        self.running_cnt = T.zeros(1, dtype=T.int32, device=self.device)

        # self.lock = threading.Lock()

    def normalize(self, v):
        return T.clamp((v - self.running_mean) / self.running_std,
                       -self.clip_range, self.clip_range).float()

    def denormalize(self, v):
        return (v * self.running_std) + self.running_mean
    
    def update_local_stats(self, new_data):
        try:
            # with self.lock:
            self.local_sum += new_data.sum(dim=0)
            self.local_sum_sq += (new_data**2).sum(dim=0)
            self.local_cnt += new_data.size(0)
        except Exception as e:
            print(f"Error during update: {e}")
    
    def update_global_stats(self):
        # with self.lock:
        # local_cnt = self.local_cnt.clone()
        # local_sum = self.local_sum.clone()
        # local_sum_sq = self.local_sum_sq.clone()

        self.running_cnt += self.local_cnt
        self.running_sum += self.local_sum
        self.running_sum_sq += self.local_sum_sq

        self.local_cnt.zero_()
        self.local_sum.zero_()
        self.local_sum_sq.zero_()

        # sync_sum, sync_sum_sq, sync_cnt = self.sync_thread_stats(
        #         local_sum, local_sum_sq, local_cnt)

        self.running_mean = self.running_sum / self.running_cnt
        tmp = self.running_sum_sq / self.running_cnt -\
            (self.running_sum / self.running_cnt)**2
        self.running_std = T.sqrt(T.maximum(self.eps**2, tmp))

    def sync_thread_stats(self, local_sum, local_sum_sq, local_cnt):
        local_sum[...] = self.mpi_average(local_sum)
        local_sum_sq[...] = self.mpi_average(local_sum_sq)
        local_cnt[...] = self.mpi_average(local_cnt)
        return local_sum, local_sum_sq, local_cnt

    def mpi_average(self, x):
        buf = T.zeros_like(x)
        MPI.COMM_WORLD.Allreduce(x.cpu().numpy(), buf.cpu().numpy(), op=MPI.SUM)
        buf = buf.float() / MPI.COMM_WORLD.Get_size()
        return buf.to(self.device)

    def get_config(self):
        return {
            "params":{
                'size':self.size,
                'eps':self.eps,
                'clip_range':self.clip_range,
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

    def save_state(self, file_path):
        T.save({
            'local_sum': self.local_sum,
            'local_sum_sq': self.local_sum_sq,
            'local_cnt': self.local_cnt,
            'running_mean': self.running_mean,
            'running_std': self.running_std,
            'running_sum': self.running_sum,
            'running_sum_sq': self.running_sum_sq,
            'running_cnt': self.running_cnt,
        }, file_path)

    @classmethod
    def load_state(cls, file_path, device='cpu'):
        state = T.load(file_path)
        normalizer = cls(size=state['running_mean'].shape, device=device)
        normalizer.local_sum = state['local_sum']
        normalizer.local_sum_sq = state['local_sum_sq']
        normalizer.local_cnt = state['local_cnt']
        normalizer.running_mean = state['running_mean']
        normalizer.running_std = state['running_std']
        normalizer.running_sum = state['running_sum']
        normalizer.running_sum_sq = state['running_sum_sq']
        normalizer.running_cnt = state['running_cnt']
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