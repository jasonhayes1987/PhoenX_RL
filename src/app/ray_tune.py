import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.wandb import WandbLoggerCallback
import os

from distributed_trainer import RayDistributedAgent
from rl_agents import PPO, ActorCritic, Reinforce

class RayTuneIntegration:
    """
    Integrates Ray Tune for hyperparameter optimization with RL agents.
    """
    def __init__(
        self,
        agent_class,
        base_config,
        search_space,
        num_workers=2,
        cpu_per_trial=1,
        gpu_per_trial=0,
        num_samples=10,
        scheduler_type="asha",
        scheduler_config=None,
        wandb_project=None,
        sync_frequency=10,
        metric="episode_reward",
        mode="max",
        local_dir="./ray_results"
    ):
        """
        Initialize the Ray Tune integration.
        
        Args:
            agent_class: The class of the agent to tune (e.g., DDPG, TD3)
            base_config: Base configuration for the agent
            search_space: Search space for hyperparameters
            num_workers: Number of parallel workers for each trial
            cpu_per_trial: CPUs to allocate per trial
            gpu_per_trial: GPUs to allocate per trial
            num_samples: Number of trial samples to generate
            scheduler_type: Scheduler type ('asha', 'pbt', 'hyperband')
            scheduler_config: Additional configurations for the scheduler
            wandb_project: W&B project name for logging
            sync_frequency: How often to synchronize gradients (in steps)
            metric: Metric to optimize
            mode: Optimization mode ('max' or 'min')
            local_dir: Directory to store results
        """
        self.agent_class = agent_class
        self.base_config = base_config
        self.search_space = search_space
        self.num_workers = num_workers
        self.cpu_per_trial = cpu_per_trial
        self.gpu_per_trial = gpu_per_trial
        self.num_samples = num_samples
        self.scheduler_type = scheduler_type
        self.scheduler_config = scheduler_config or {}
        self.wandb_project = wandb_project
        self.sync_frequency = sync_frequency
        self.metric = metric
        self.mode = mode
        self.local_dir = local_dir
        
        # Initialize Ray if not already started
        if not ray.is_initialized():
            ray.init()
    
    def _get_scheduler(self):
        """Get the appropriate scheduler based on the configuration."""
        if self.scheduler_type == "asha":
            return ASHAScheduler(
                time_attr='training_iteration',
                metric=self.metric,
                mode=self.mode,
                max_t=100,
                grace_period=10,
                reduction_factor=2,
                **self.scheduler_config
            )
        elif self.scheduler_type == "pbt":
            return PopulationBasedTraining(
                time_attr='training_iteration',
                metric=self.metric,
                mode=self.mode,
                perturbation_interval=10,
                **self.scheduler_config
            )
        else:
            return None
    
    def _get_callbacks(self):
        """Get callbacks for Ray Tune."""
        callbacks = []
        
        if self.wandb_project:
            callbacks.append(
                WandbLoggerCallback(
                    project=self.wandb_project,
                    log_config=True,
                    api_key=os.environ.get("WANDB_API_KEY")
                )
            )
        
        return callbacks
    
    def _train_function(self, config):
        """
        Training function for Ray Tune.
        
        Args:
            config: Configuration dict that includes hyperparameters
        """
        # Create a copy of base_config and update with tune params
        agent_config = self.base_config.copy()
        agent_config.update(config)
        
        # Create distributed agent
        distributed_agent = RayDistributedAgent(
            self.agent_class,
            agent_config,
            num_workers=self.num_workers,
            sync_frequency=self.sync_frequency,
            device=config.get("device", "cpu"),
            wandb_project=None  # We handle W&B through Tune
        )
        
        # Extract training parameters
        train_kwargs = {
            "num_episodes": config.get("num_episodes", 1000),
            "render_freq": 0  # No rendering during tuning
        }
        
        # Add specific parameters for different agent types
        if issubclass(self.agent_class, PPO):
            train_kwargs.update({
                "trajectory_length": config.get("trajectory_length", 128),
                "batch_size": config.get("batch_size", 64),
                "learning_epochs": config.get("learning_epochs", 4),
                "num_envs": config.get("num_envs", 4)
            })
        elif issubclass(self.agent_class, (ActorCritic, Reinforce)):
            train_kwargs.update({
                "num_envs": config.get("num_envs", 4)
            })
        
        # Train the agent and collect results
        results = distributed_agent.train(**train_kwargs)
        
        # Extract and report metrics
        metrics = results.get("metrics", {})
        for i in range(100):  # Report metrics for each training iteration
            if i >= len(metrics.get(self.metric, [])):
                break
                
            tune.report(
                training_iteration=i,
                **{k: v[i] if i < len(v) else v[-1] for k, v in metrics.items()}
            )
    
    def run(self):
        """Run the hyperparameter tuning experiment."""
        scheduler = self._get_scheduler()
        callbacks = self._get_callbacks()
        
        # Run the tuning
        analysis = tune.run(
            self._train_function,
            config=self.search_space,
            resources_per_trial={
                "cpu": self.cpu_per_trial,
                "gpu": self.gpu_per_trial
            },
            num_samples=self.num_samples,
            scheduler=scheduler,
            local_dir=self.local_dir,
            callbacks=callbacks,
            progress_reporter=tune.CLIReporter(
                metric_columns=[self.metric, "training_iteration"]
            )
        )
        
        # Print the best configuration
        print("Best config:", analysis.best_config)
        
        # Return the results
        return {
            "best_config": analysis.best_config,
            "best_result": analysis.best_result,
            "analysis": analysis
        }