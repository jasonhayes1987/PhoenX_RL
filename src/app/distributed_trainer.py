import ray
import asyncio
import torch as T
from typing import Dict, List
import logging
from rl_agents import Agent
from agent_utils import load_agent_from_config, convert_to_distributed_callbacks
from buffer import Buffer
from env_wrapper import EnvWrapper
from logging_config import get_logger
from rl_callbacks import RayWandbCallback
# import ray.logger as logger

# Use correct Ray logger import
# import logging
# logger = logging.getLogger("ray")



@ray.remote(num_cpus=1)
class SharedBuffer:
    """
    Shared buffer for distributed training
    """
    def __init__(self, buffer_config: Dict, log_level='info'):
        try:
            self.logger = get_logger(__name__, level=log_level)
            self.logger.info(f"Initializing SharedBuffer with config: {buffer_config}")
            self.buffer_config = buffer_config
            env_wrapper = EnvWrapper.from_json(buffer_config['config']['env'])
            self.buffer_config['config']['env'] = env_wrapper
            self.buffer = Buffer.create_instance(buffer_config['class_name'], **buffer_config['config'])
            self.logger.info("SharedBuffer initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing shared buffer: {e}", exc_info=True)
            raise e 
    
    def add(self, *args, **kwargs):
        """Add data to the buffer"""
        try:
            return self.buffer.add(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error adding to shared buffer: {e}", exc_info=True)
            raise
    
    def sample(self, batch_size: int):
        """Sample data from the buffer"""
        try:
            return self.buffer.sample(batch_size)
        except Exception as e:
            self.logger.error(f"Error sampling from shared buffer: {e}", exc_info=True)
            raise
    
    def update_priorities(self, indices: T.Tensor, priorities: T.Tensor):
        """Update priorities of the sampled data"""
        try:
            return self.buffer.update_priorities(indices, priorities)
        except Exception as e:
            self.logger.error(f"Error updating priorities in shared buffer: {e}", exc_info=True)
            raise
    
    def get_sum_tree_capacity(self):
        """Get the capacity of the sum tree"""
        try:
            return self.buffer.sum_tree.capacity
        except Exception as e:
            self.logger.error(f"Error getting sum tree capacity: {e}", exc_info=True)
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
    def __init__(self, shared_buffer: SharedBuffer, prioritized: bool = False, log_level='info'):
        try:
            self.logger = get_logger(__name__, level=log_level)
            self.logger.info(f"Initializing BufferWrapper with prioritized={prioritized}")
            self.shared_buffer = shared_buffer
            self.prioritized = prioritized
            proportional = ray.get(shared_buffer.get_config.remote())['config']['priority'] == 'proportional' if prioritized else False
            self._sum_tree = SumTreeWrapper(shared_buffer) if proportional else None
            self.logger.info("BufferWrapper initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing buffer wrapper: {e}", exc_info=True)
            raise e

    def add(self, *args, **kwargs):
        """Add data to the buffer"""
        try:
            return ray.get(self.shared_buffer.add.remote(*args, **kwargs))
        except Exception as e:
            self.logger.error(f"Error adding to buffer wrapper: {e}", exc_info=True)
            raise
    
    def sample(self, batch_size: int):
        """Sample data from the buffer"""
        try:
            return ray.get(self.shared_buffer.sample.remote(batch_size))
        except Exception as e:
            self.logger.error(f"Error sampling from buffer wrapper: {e}", exc_info=True)
            raise
    
    def update_priorities(self, indices: T.Tensor, priorities: T.Tensor):
        """Update priorities of the sampled data"""
        try:
            return ray.get(self.shared_buffer.update_priorities.remote(indices, priorities))
        except Exception as e:
            self.logger.error(f"Error updating priorities in buffer wrapper: {e}", exc_info=True)
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
    def __init__(self, shared_buffer: SharedBuffer, log_level='info'):
        try:
            self.logger = get_logger(__name__, level=log_level)
            self.logger.info("Initializing SumTreeWrapper")
            self._shared_buffer = shared_buffer
            self._capacity = None
            self.logger.info("SumTreeWrapper initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing sum tree wrapper: {e}", exc_info=True)
            raise e

    @property
    def capacity(self):
        """Get the capacity of the sum tree"""
        try:
            if self._capacity is None:
                self._capacity = ray.get(self._shared_buffer.get_sum_tree_capacity.remote())
            return self._capacity
        except Exception as e:
            self.logger.error(f"Error getting sum tree capacity: {e}", exc_info=True)
            raise

# @ray.remote
# class GradientSynchronizer:
#     def __init__(self, num_workers, log_level='info'):
#         try:
#             self.logger = get_logger(__name__, level=log_level)
#             self.logger.info(f"Initializing GradientSynchronizer with {num_workers} workers")
#             self.num_workers = num_workers
#             self.events = {}
#             self.gradients = {}
#             self.averaged_gradients = {}
#             self.counters = {}
#             self.logger.info("GradientSynchronizer initialized successfully")
#         except Exception as e:
#             self.logger.error(f"Error initializing gradient synchronizer: {e}", exc_info=True)
#             raise e

#     async def submit_gradients(self, model_key, gradients):
#         """
#         Submit gradients for a model and waits for all workers to submit
#         gradients before returning the averaged gradients
#         """
#         try:
#             if model_key not in self.gradients:
#                 self.gradients[model_key] = []
#                 self.counters[model_key] = 0
#                 self.events[model_key] = asyncio.Event()

#             self.gradients[model_key].append(gradients)
#             self.counters[model_key] += 1
#             self.logger.debug(f"Received gradients from worker {self.counters[model_key]}/{self.num_workers} for model {model_key}")
#             self.logger.debug(f"Received gradients: {gradients}")
#             self.logger.debug(f"Gradients: {self.gradients[model_key]}")

#             if self.counters[model_key] == self.num_workers:
#                 self.averaged_gradients[model_key] = self._compute_average(self.gradients[model_key])
#                 self.events[model_key].set()
#                 self.gradients[model_key] = []
#                 self.counters[model_key] = 0
#                 self.logger.debug(f"Averaged gradients for model {model_key}")
#                 self.logger.debug(f"Averaged gradients: {self.averaged_gradients[model_key]}")
#             await self.events[model_key].wait()
#             return self.averaged_gradients[model_key]
#         except Exception as e:
#             self.logger.error(f"Error in submit_gradients: {e}", exc_info=True)
#             raise
    
#     def _compute_average(self, gradients):
#         """Average gradients for each parameter across workers"""
#         try:
#             return [T.stack([g[i] for g in gradients]).mean(dim=0) 
#                     for i in range(len(gradients[0]))]
#         except Exception as e:
#             self.logger.error(f"Error computing gradient average: {e}", exc_info=True)
#             raise

#     def get_config(self):
#         return {
#             'num_workers': self.num_workers,
#         }
@ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self, agent: Agent, buffer: BufferWrapper = None, log_level='info'):
        self.agent = agent
        self.buffer = buffer
        # Replace buffer from agent with buffer wrapper if using shared buffer
        if self.buffer:
            self.agent.replay_buffer = self.buffer
        self.logger = get_logger(__name__, level=log_level)
        self.log_level = logging.getLevelName(self.logger.getEffectiveLevel()).lower()

        # Verify agent is on correct device (should be CUDA if available)
        self.device = agent.device
        self.logger.info(f"Learner agent using device: {self.device}")

        # Set internal step counter(tracks num times learn is called)
        self._learn_step = 0

        # Set step counter on agent to 0
        self.agent._step = 0

    def learn(self, step: int, run_number:str=None, gradients:Dict[str, List[T.Tensor]] = None):
        """Update the agent's models using gradients"""
        self._learn_step += 1
        
        try:
            if gradients:
                self.agent._distributed_learn(step, run_number, gradients)
            else:
                self.agent._distributed_learn(step, run_number)
            
        except Exception as e:
            self.logger.error(f"Error in learn: {e}", exc_info=True)
            raise

    def get_parameters(self):
        """Get the parameters of the agent"""
        return self.agent.get_parameters()
    
    def reset(self):
        """Reset the step counter on the agent"""
        self.agent._step = 0
        self._learn_step = 0

class DistributedAgents:
    def __init__(self, agent_config, num_workers, learner_device: str = None, worker_device: str = None, log_level='info'):
        """Initialize distributed agents with the given configuration"""
        try:
            self.logger = get_logger(__name__, level=log_level)
            self.logger.info(f"Initializing DistributedAgents with {num_workers} workers")
            self.agent_config = agent_config
            self.num_workers = num_workers
            self.workers = []
            self.learner_device = learner_device
            self.worker_device = worker_device
            
            # # Initialize GradientSynchronizer
            # try:
            #     self.logger.info("Creating GradientSynchronizer actor")
            #     self.gradient_synchronizer = GradientSynchronizer.remote(num_workers, log_level)
            #     self.logger.info("GradientSynchronizer actor created successfully")
            # except Exception as e:
            #     self.logger.error(f"Failed to create GradientSynchronizer actor: {e}", exc_info=True)
            #     raise
            
            # Initialize SharedBuffer if configured
            if 'replay_buffer' in agent_config:
                try:
                    self.logger.info("Creating SharedBuffer actor")
                    self.shared_buffer = SharedBuffer.remote(agent_config['replay_buffer'], log_level)
                    self.logger.info("SharedBuffer actor created successfully")
                except Exception as e:
                    self.logger.error(f"Failed to create SharedBuffer actor: {e}", exc_info=True)
                    raise
            else:
                self.shared_buffer = None
                self.logger.info("No replay buffer configured")
            
            # Setup workers
            self.setup_workers(log_level)
            self.logger.info("DistributedAgents initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing DistributedAgents: {e}", exc_info=True)
            raise

    def setup_workers(self, log_level='info'):
        try:
            # Initialize Ray and get available resources
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, log_to_driver=True, dashboard_host='127.0.0.1', dashboard_port=8265)
            
            RESOURCES = ray.available_resources()
            CPUS = RESOURCES['CPU'] - 4 # Leave 2 for buffer class and learner and 2 as a buffer zone
            GPUS = RESOURCES.get('GPU', 0)
            self.logger.info(f'Ray initialized with {CPUS} CPUs and {GPUS} GPUs available')
            num_cpus_per_worker = max(1, CPUS // self.num_workers)
            
            # Create base agent - explicitly force to CPU
            self.logger.info(f'Creating base agent with device=cpu for safe cloning')
            base_agent = load_agent_from_config(self.agent_config, load_weights=False)
            
            # Create a reference agent on CPU for cloning
            # self.logger.info(f'Creating CPU reference agent for workers')
            # cpu_agent = base_agent.clone(device='cpu')
            
            # Create a GPU agent for the learner if GPUs are available
            # learner_device = 'cuda' if GPUS > 0 else 'cpu'
            self.logger.info(f'Creating Learner agent with device={self.learner_device}')
            learner_agent = base_agent.clone(device=self.learner_device)
            # Convert learner agent WandbCallback to DistributedCallback
            learner_agent.callbacks = convert_to_distributed_callbacks(learner_agent.callbacks, "learner", 0)
            # Initialize DistributedCallback with the learner agent
            for callback in learner_agent.callbacks:
                if isinstance(callback, RayWandbCallback):
                    learner_agent._config = callback._config(learner_agent)
            # Initialize Learner with its own copy of the agent
            if self.shared_buffer:
                prioritized = self.agent_config['replay_buffer']['class_name'] == 'PrioritizedReplayBuffer'
                buffer = BufferWrapper(self.shared_buffer, prioritized, log_level)
            else:
                buffer = None
            self.learner = Learner.options(num_cpus=1, num_gpus=(1 if GPUS > 0 else 0)).remote(
                base_agent, buffer, log_level)
            self.logger.info(f'Learner initialized successfully')
            
            # Get the learner's ID in a way that works with current Ray versions
            # learner_id = ray.util.get_actor_name(self.learner) or "global_learner"
            # self.logger.info(f'Using learner ID: {learner_id}')
            
            # Register the actor with a consistent name if not already named
            
            # ray.util.register_actor(self.learner, learner_id)
            # self.logger.info(f'Registered learner with ID: {learner_id}')
            
            self.logger.info(f'Setting up {self.num_workers} workers with {num_cpus_per_worker} CPUs each')
            
            for i in range(self.num_workers):
                try:
                    if self.shared_buffer:
                        prioritized = self.agent_config['replay_buffer']['class_name'] == 'PrioritizedReplayBuffer'
                        buffer = BufferWrapper(self.shared_buffer, prioritized, log_level)
                    else:
                        buffer = None
                    
                    # Create a fresh CPU-only clone for each worker
                    self.logger.info(f'Creating worker {i} agent with device=cpu')
                    worker_agent = learner_agent.clone(device='cpu')
                    
                    self.logger.info(f'Creating Worker {i} actor')
                    worker = Worker.options(
                        num_cpus=num_cpus_per_worker,
                        num_gpus=0,
                        max_restarts=3,
                        max_task_retries=3
                    ).remote(worker_agent, self.learner, buffer, i, log_level)
                    
                    self.workers.append(worker)
                except Exception as e:
                    self.logger.error(f'Failed to create Worker {i}: {e}', exc_info=True)
                    raise
        except Exception as e:
            self.logger.error(f"Error setting up workers: {e}", exc_info=True)
            raise

    def train(self, sync_interval: int = 1, **kwargs):
        """Train the agent in a distributed manner"""
        try:
            self.logger.info("Starting distributed training")
            
            # Verify all workers are still alive
            for i, worker in enumerate(self.workers):
                try:
                    ray.get(worker.get_agent_config.remote())
                except Exception as e:
                    self.logger.error(f"Worker {i} is not responding: {e}", exc_info=True)
                    raise
            
            # Submit training tasks to all workers
            futures = []
            for i, worker in enumerate(self.workers):
                try:
                    future = worker.train.remote(sync_interval=sync_interval, **kwargs)
                    futures.append(future)
                    self.logger.info(f"Submitted training task to worker {i}")
                except Exception as e:
                    self.logger.error(f"Failed to submit training task to worker {i}: {e}", exc_info=True)
                    raise
            
            self.logger.info(f"Submitted training tasks to {len(futures)} workers")
            return futures
        except Exception as e:
            self.logger.error(f"Error in distributed training: {e}", exc_info=True)
            raise
    
    def get_config(self):
        return {
            'agent_config': self.agent_config,
            'num_workers': self.num_workers,
        }

@ray.remote(num_cpus=1, num_gpus=0)
class Worker:
    def __init__(self, agent: Agent, learner: Learner, buffer: BufferWrapper = None, id: int = None, log_level='info'):
        try:
            self.logger = get_logger(f"Worker {id}", level=log_level)
            self.logger.info("Initializing Worker")
            self.worker_id = id
            self.agent = agent
            # Log the device information to debug
            self.logger.info(f"Worker agent using device: {self.agent.device}")
            self.learner = learner
            self.logger.info(f"Successfully obtained learner reference")
            self.buffer = buffer
            self.agent._distributed = True
            
            # If RayWandbCallback, set worker id to passed id
            for callback in self.agent.callbacks:
                if isinstance(callback, RayWandbCallback):
                    callback.worker_id = self.worker_id
                    callback.role = "worker"
                    # callback.is_main_worker = (self.worker_id == 0)

            # Set the _distributed_learn function on the agent to point to Learner
            if self.buffer:
                self.agent._distributed_learn = lambda step, run_number: self.learner.learn.remote(step, run_number)
            else:
                self.agent._distributed_learn = lambda step, run_number, gradients: self.learner.learn.remote(step, run_number, gradients)

            # Set the get_parameters function on the agent to point to Learner (blocking)
            self.agent.get_parameters = lambda: ray.get(self.learner.get_parameters.remote())

            # Replace buffer from agent with buffer wrapper if using shared buffer
            if self.buffer:
                self.agent.replay_buffer = self.buffer
            self.logger.info("Worker initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing worker: {e}", exc_info=True)
            raise e

    def train(self, **kwargs):
        try:
            self.logger.info("Starting worker training")
            self.agent.train(**kwargs)
            self.logger.info("Worker training completed")
        except Exception as e:
            self.logger.error(f"Error in worker training: {e}", exc_info=True)
            raise

    def get_agent_config(self):
        return self.agent.get_config()
    
    def get_gradient_synchronizer(self):
        return self.gradient_synchronizer
    
