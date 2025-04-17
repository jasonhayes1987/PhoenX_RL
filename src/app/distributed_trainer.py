import ray
import asyncio
import torch as T
from typing import Dict, List
import logging
from agent_utils import load_agent_from_config
from buffer import Buffer
from env_wrapper import EnvWrapper
from logging_config import logger as rl_logger



@ray.remote(num_cpus=1)
class SharedBuffer:
    """
    Shared buffer for distributed training
    """
    def __init__(self, buffer_config: Dict):
        try:
            rl_logger.info(f"Initializing SharedBuffer with config: {buffer_config}")
            self.buffer_config = buffer_config
            env_wrapper = EnvWrapper.from_json(buffer_config['config']['env'])
            self.buffer_config['config']['env'] = env_wrapper
            self.buffer = Buffer.create_instance(buffer_config['class_name'], **buffer_config['config'])
            rl_logger.info("SharedBuffer initialized successfully")
        except Exception as e:
            rl_logger.error(f"Error initializing shared buffer: {e}", exc_info=True)
            raise e 
    
    def add(self, *args, **kwargs):
        """Add data to the buffer"""
        try:
            return self.buffer.add(*args, **kwargs)
        except Exception as e:
            rl_logger.error(f"Error adding to shared buffer: {e}", exc_info=True)
            raise
    
    def sample(self, batch_size: int):
        """Sample data from the buffer"""
        try:
            return self.buffer.sample(batch_size)
        except Exception as e:
            rl_logger.error(f"Error sampling from shared buffer: {e}", exc_info=True)
            raise
    
    def update_priorities(self, indices: T.Tensor, priorities: T.Tensor):
        """Update priorities of the sampled data"""
        try:
            return self.buffer.update_priorities(indices, priorities)
        except Exception as e:
            rl_logger.error(f"Error updating priorities in shared buffer: {e}", exc_info=True)
            raise
    
    def get_sum_tree_capacity(self):
        """Get the capacity of the sum tree"""
        try:
            return self.buffer.sum_tree.capacity
        except Exception as e:
            rl_logger.error(f"Error getting sum tree capacity: {e}", exc_info=True)
            raise
            
    def priorities(self):
        """Get the priorities of the buffer"""
        return self.buffer.priorities
    
    def beta(self):
        """Get the beta of the buffer"""
        return self.buffer.beta

    # @property
    def buffer_size(self):
        """Get the size of the buffer"""
        return self.buffer.buffer_size
    
    # @property
    def counter(self):
        """Get the counter of the buffer"""
        return self.buffer.counter
    
    # @property
    def device(self):
        """Get the device of the buffer"""
        return self.buffer.device
    
    def get_config(self):
        """Get the config of the buffer"""
        return self.buffer.get_config()

class BufferWrapper:
    """
    Wrapper for Buffer class to interface with a shared buffer for distributed training
    """
    def __init__(self, shared_buffer: SharedBuffer, prioritized: bool = False):
        try:
            rl_logger.info(f"Initializing BufferWrapper with prioritized={prioritized}")
            self.shared_buffer = shared_buffer
            self.prioritized = prioritized
            proportional = ray.get(shared_buffer.get_config.remote())['config']['priority'] == 'proportional' if prioritized else False
            self._sum_tree = SumTreeWrapper(shared_buffer) if proportional else None
            rl_logger.info("BufferWrapper initialized successfully")
        except Exception as e:
            rl_logger.error(f"Error initializing buffer wrapper: {e}", exc_info=True)
            raise e

    def add(self, *args, **kwargs):
        """Add data to the buffer"""
        try:
            return ray.get(self.shared_buffer.add.remote(*args, **kwargs))
        except Exception as e:
            rl_logger.error(f"Error adding to buffer wrapper: {e}", exc_info=True)
            raise
    
    def sample(self, batch_size: int):
        """Sample data from the buffer"""
        try:
            return ray.get(self.shared_buffer.sample.remote(batch_size))
        except Exception as e:
            rl_logger.error(f"Error sampling from buffer wrapper: {e}", exc_info=True)
            raise
    
    def update_priorities(self, indices: T.Tensor, priorities: T.Tensor):
        """Update priorities of the sampled data"""
        try:
            return ray.get(self.shared_buffer.update_priorities.remote(indices, priorities))
        except Exception as e:
            rl_logger.error(f"Error updating priorities in buffer wrapper: {e}", exc_info=True)
            raise
    
    def get_config(self):
        """Get the config of the buffer"""
        return ray.get(self.shared_buffer.get_config.remote())
    
    @property
    def buffer_size(self):
        """Get the size of the buffer"""
        return ray.get(self.shared_buffer.buffer_size.remote())
    
    @property
    def counter(self):
        """Get the counter of the buffer"""
        return ray.get(self.shared_buffer.counter.remote())
    
    @property
    def device(self):
        """Get the device of the buffer"""
        return ray.get(self.shared_buffer.device.remote())
    
    @property
    def sum_tree(self):
        """Get the sum tree of the buffer"""
        return self._sum_tree
    
    @property
    def priorities(self):
        """Get the priorities of the buffer"""
        return ray.get(self.shared_buffer.priorities.remote())
    
    @property
    def beta(self):
        """Get the beta of the buffer"""
        return ray.get(self.shared_buffer.beta.remote())

class SumTreeWrapper:
    """
    Wrapper for SumTree class to interface with a shared sum tree for distributed training
    """
    def __init__(self, shared_buffer: SharedBuffer):
        try:
            rl_logger.info("Initializing SumTreeWrapper")
            self._shared_buffer = shared_buffer
            self._capacity = None
            rl_logger.info("SumTreeWrapper initialized successfully")
        except Exception as e:
            rl_logger.error(f"Error initializing sum tree wrapper: {e}", exc_info=True)
            raise e

    @property
    def capacity(self):
        """Get the capacity of the sum tree"""
        try:
            if self._capacity is None:
                self._capacity = ray.get(self._shared_buffer.get_sum_tree_capacity.remote())
            return self._capacity
        except Exception as e:
            rl_logger.error(f"Error getting sum tree capacity: {e}", exc_info=True)
            raise

@ray.remote
class GradientSynchronizer:
    def __init__(self, num_workers):
        try:
            rl_logger.info(f"Initializing GradientSynchronizer with {num_workers} workers")
            self.num_workers = num_workers
            self.events = {}
            self.gradients = {}
            self.averaged_gradients = {}
            self.counters = {}
            rl_logger.info("GradientSynchronizer initialized successfully")
        except Exception as e:
            rl_logger.error(f"Error initializing gradient synchronizer: {e}", exc_info=True)
            raise e

    async def submit_gradients(self, model_key, gradients):
        """
        Submit gradients for a model and waits for all workers to submit
        gradients before returning the averaged gradients
        """
        try:
            if model_key not in self.gradients:
                self.gradients[model_key] = []
                self.counters[model_key] = 0
                self.events[model_key] = asyncio.Event()

            self.gradients[model_key].append(gradients)
            self.counters[model_key] += 1
            rl_logger.debug(f"Received gradients from worker {self.counters[model_key]}/{self.num_workers} for model {model_key}")
            
            if self.counters[model_key] == self.num_workers:
                self.averaged_gradients[model_key] = self._compute_average(self.gradients[model_key])
                self.events[model_key].set()
                self.gradients[model_key] = []
                self.counters[model_key] = 0
                rl_logger.info(f"Averaged gradients for model {model_key}")

            await self.events[model_key].wait()
            return self.averaged_gradients[model_key]
        except Exception as e:
            rl_logger.error(f"Error in submit_gradients: {e}", exc_info=True)
            raise
    
    def _compute_average(self, gradients):
        """Average gradients for each parameter across workers"""
        try:
            return [T.stack([g[i] for g in gradients]).mean(dim=0) 
                    for i in range(len(gradients[0]))]
        except Exception as e:
            rl_logger.error(f"Error computing gradient average: {e}", exc_info=True)
            raise

    def get_config(self):
        return {
            'num_workers': self.num_workers,
        }

class DistributedAgents:
    def __init__(self, agent_config, num_workers):
        """Initialize distributed agents with the given configuration"""
        try:
            rl_logger.info(f"Initializing DistributedAgents with {num_workers} workers")
            self.agent_config = agent_config
            self.num_workers = num_workers
            self.workers = []
            
            # Initialize GradientSynchronizer
            try:
                rl_logger.info("Creating GradientSynchronizer actor")
                self.gradient_synchronizer = GradientSynchronizer.remote(num_workers)
                rl_logger.info("GradientSynchronizer actor created successfully")
            except Exception as e:
                rl_logger.error(f"Failed to create GradientSynchronizer actor: {e}", exc_info=True)
                raise
            
            # Initialize SharedBuffer if configured
            if 'replay_buffer' in agent_config:
                try:
                    rl_logger.info("Creating SharedBuffer actor")
                    self.shared_buffer = SharedBuffer.remote(agent_config['replay_buffer'])
                    rl_logger.info("SharedBuffer actor created successfully")
                except Exception as e:
                    rl_logger.error(f"Failed to create SharedBuffer actor: {e}", exc_info=True)
                    raise
            else:
                self.shared_buffer = None
                rl_logger.info("No replay buffer configured")
            
            # Setup workers
            self.setup_workers()
            rl_logger.info("DistributedAgents initialized successfully")
        except Exception as e:
            rl_logger.error(f"Error initializing DistributedAgents: {e}", exc_info=True)
            raise

    def setup_workers(self):
        try:
            # Initialize Ray and get available resources
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            RESOURCES = ray.available_resources()
            CPUS = RESOURCES['CPU']
            GPUS = RESOURCES['GPU']
            rl_logger.info(f'Ray initialized with {CPUS} CPUs and {GPUS} GPUs available')
            num_cpus_per_worker = CPUS // self.num_workers
            num_gpus_per_worker = round(GPUS / self.num_workers, 2)
            rl_logger.info(f'Setting up {self.num_workers} workers with {num_cpus_per_worker} CPUs and {num_gpus_per_worker} GPUs each')
            for i in range(self.num_workers):
                try:
                    if self.shared_buffer:
                        prioritized = self.agent_config['replay_buffer']['class_name'] == 'PrioritizedReplayBuffer'
                        buffer = BufferWrapper(self.shared_buffer, prioritized)
                    else:
                        buffer = None
                    
                    rl_logger.info(f'Creating Worker {i} actor')
                    worker = Worker.options(
                        num_cpus=1, #num_cpus_per_worker, 
                        num_gpus=num_gpus_per_worker,
                        max_restarts=3,  # Allow some retries
                        max_task_retries=3
                    ).remote(self.agent_config, self.gradient_synchronizer, buffer)
                    
                    # # Verify worker creation
                    # try:
                    #     ray.get(worker.get_agent_config.remote())
                    #     rl_logger.info(f'Worker {i} actor created and verified successfully')
                    # except Exception as e:
                    #     rl_logger.error(f'Worker {i} actor creation verification failed: {e}', exc_info=True)
                    #     raise
                    
                    self.workers.append(worker)
                except Exception as e:
                    rl_logger.error(f'Failed to create Worker {i}: {e}', exc_info=True)
                    raise
        except Exception as e:
            rl_logger.error(f"Error setting up workers: {e}", exc_info=True)
            raise

    def train(self, **kwargs):
        """Train the agent in a distributed manner"""
        try:
            rl_logger.info("Starting distributed training")
            
            # Verify all workers are still alive
            for i, worker in enumerate(self.workers):
                try:
                    ray.get(worker.get_agent_config.remote())
                except Exception as e:
                    rl_logger.error(f"Worker {i} is not responding: {e}", exc_info=True)
                    raise
            
            # Submit training tasks to all workers
            futures = []
            for i, worker in enumerate(self.workers):
                try:
                    future = worker.train.remote(**kwargs)
                    futures.append(future)
                    rl_logger.info(f"Submitted training task to worker {i}")
                except Exception as e:
                    rl_logger.error(f"Failed to submit training task to worker {i}: {e}", exc_info=True)
                    raise
            
            rl_logger.info(f"Submitted training tasks to {len(futures)} workers")
            return futures
        except Exception as e:
            rl_logger.error(f"Error in distributed training: {e}", exc_info=True)
            raise
    
    def get_config(self):
        return {
            'agent_config': self.agent_config,
            'num_workers': self.num_workers,
        }

@ray.remote(num_cpus=1, num_gpus=1)
class Worker:
    def __init__(self, config: Dict, gradient_synchronizer: GradientSynchronizer, buffer: BufferWrapper = None):
        try:
            rl_logger.info("Initializing Worker")
            self.agent = load_agent_from_config(config)
            self.gradient_synchronizer = gradient_synchronizer
            self.buffer = buffer
            self.agent._distributed = True
            self.loop = asyncio.get_event_loop()

            # Define async function to submit gradients to the gradient synchronizer
            async def sync_gradients(model_key, gradients):
                try:
                    avg_gradient = await self.gradient_synchronizer.submit_gradients.remote(model_key, gradients)
                    return avg_gradient
                except Exception as e:
                    rl_logger.error(f"Error in sync_gradients: {e}", exc_info=True)
                    raise
            
            # Set the async function as a blocking method on the agent 
            self.agent._sync_gradients = lambda model_key, gradients: self.loop.run_until_complete(sync_gradients(model_key, gradients))

            # Replace buffer from agent with buffer wrapper if using shared buffer
            if self.buffer:
                self.agent.replay_buffer = self.buffer
            rl_logger.info("Worker initialized successfully")
        except Exception as e:
            rl_logger.error(f"Error initializing worker: {e}", exc_info=True)
            raise e

    def train(self, **kwargs):
        try:
            rl_logger.info("Starting worker training")
            self.agent.train(**kwargs)
            rl_logger.info("Worker training completed")
        except Exception as e:
            rl_logger.error(f"Error in worker training: {e}", exc_info=True)
            raise

    def get_agent_config(self):
        return self.agent.get_config()
    
    def get_gradient_synchronizer(self):
        return self.gradient_synchronizer
    
    def check_sync_gradients_set(self):
        """Check if _sync_gradients attribute is correctly set on the agent"""
        try:
            has_attr = hasattr(self.agent, '_sync_gradients')
            is_callable = callable(self.agent._sync_gradients) if has_attr else False
            rl_logger.debug(f"Sync gradients check: has_attribute={has_attr}, is_callable={is_callable}")
            return {"has_attribute": has_attr, "is_callable": is_callable}
        except Exception as e:
            rl_logger.error(f"Error checking sync gradients: {e}", exc_info=True)
            raise
    
