import ray
from ray import tune
from ray.tune import Tuner
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air.integrations.wandb import WandbLoggerCallback
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from agent_utils import get_agent_class_from_type

def rl_trainable(config):
    # Create vectorized environment
    def make_env():
        return gym.make(config['env_name'])
    
    env = SyncVectorEnv([make_env for _ in range(config.get('num_envs', 1))])
    
    # Get agent class
    agent_class = get_agent_class_from_type(config['algorithm'])
    
    # Common params (pulled directly from config)
    agent_kwargs = {
        'env': env,
        'learning_rate': config['lr'],
        'gamma': config['gamma'],
        'batch_size': config['batch_size'],
        'buffer_size': config['buffer_size'],
        'per_alpha': config['per_alpha'],
        'per_beta': config['per_beta'],
    }
    
    # Agent-specific params (directly from config)
    if config['algorithm'] in ['DDPG', 'TD3', 'SAC']:
        agent_kwargs.update({
            'replay_buffer': config.get('replay_buffer'),  # Assume provided
            'tau': config['tau'],
            'warmup': config['warmup'],
            'N': config['N'],
        })
        if config['algorithm'] == 'DDPG':
            agent_kwargs['action_epsilon'] = config['action_epsilon']
        elif config['algorithm'] == 'TD3':
            agent_kwargs['actor_update_delay'] = config['actor_update_delay']
            agent_kwargs['target_noise_clip'] = config['target_noise_clip']
        elif config['algorithm'] == 'SAC':
            agent_kwargs['alpha'] = config['alpha']
            agent_kwargs['auto_entropy_tuning'] = config['auto_entropy_tuning']
            agent_kwargs['alpha_lr'] = config['alpha_lr']
    elif config['algorithm'] == 'PPO':
        agent_kwargs.update({
            'gae_coefficient': config['gae_coefficient'],
            'policy_clip': config['policy_clip'],
            'value_clip': config['value_clip'],
            'value_loss_coefficient': config['value_loss_coefficient'],
            'entropy_coefficient': config['entropy_coefficient'],
            'normalize_advantages': config['normalize_advantages'],
            'grad_clip': config['grad_clip'],
        })
    elif config['algorithm'] == 'ActorCritic':
        agent_kwargs.update({
            'policy_trace_decay': config['policy_trace_decay'],
            'value_trace_decay': config['value_trace_decay'],
        })
    elif config['algorithm'] == 'Reinforce':
        pass  # Add if needed
    elif config['algorithm'] == 'HER':
        base_algorithm = config['base_algorithm']
        base_class = get_agent_class_from_type(base_algorithm)
        base_kwargs = {}  # Build from config similarly
        base_agent = base_class(**base_kwargs)
        agent_kwargs.update({
            'agent': base_agent,
            'strategy': config['strategy'],
            'tolerance': config['tolerance'],
            'num_goals': config['num_goals'],
        })
    
    agent = agent_class(**agent_kwargs)
    
    for iteration in range(config['max_iterations']):
        metrics = agent.train_step()  # Assume train_step returns metrics dict
        ray.train.report(metrics)
    
    agent.save(config['save_dir'])
    return metrics

def run_ray_tune_sweep(user_config):
    ray.init(ignore_reinit_error=True)
    
    algorithm = user_config['algorithm']
    
    # Helper function to add param to space flexibly
    def add_param(space, name, default_min=None, default_max=None, default=None, is_log=False, is_int=False, choices=None):
        if f'{name}_min' in user_config and f'{name}_max' in user_config:
            if is_log:
                space[name] = tune.loguniform(user_config[f'{name}_min'], user_config[f'{name}_max'])
            elif is_int:
                space[name] = tune.randint(user_config[f'{name}_min'], user_config[f'{name}_max'])
            else:
                space[name] = tune.uniform(user_config[f'{name}_min'], user_config[f'{name}_max'])
        elif choices and f'{name}_choices' in user_config:
            space[name] = tune.choice(user_config[f'{name}_choices'])
        elif name in user_config:
            space[name] = user_config[name]  # Fixed value
        elif default is not None:
            space[name] = default
        elif choices:
            space[name] = tune.choice(choices)
        elif default_min is not None and default_max is not None:
            if is_log:
                space[name] = tune.loguniform(default_min, default_max)
            elif is_int:
                space[name] = tune.randint(default_min, default_max)
            else:
                space[name] = tune.uniform(default_min, default_max)
    
    param_space = {
        'env_name': user_config['env_name'],
        'algorithm': algorithm,
        'max_iterations': user_config['max_iterations'],
        'device': user_config['device'],
        'save_dir': user_config.get('save_dir', 'models/'),
    }
    
    # Add common params with user_config overrides
    add_param(param_space, 'discount', default_min=0.9, default_max=0.999)
    add_param(param_space, 'num_envs', choices=[1, 4, 8])

    # Set defaults
    layer_choices = ['dense', 'conv', 'pool', 'dropout', 'batchnorm1d', 'batchnorm2d',
                     'layernorm', 'flatten', 'relu', 'leakyrelu', 'tanh']
    weight_choices = ['kaiming_uniform', 'kaiming_normal', 'xavier_uniform', 'xavier_normal', 'truncated_normal', 'uniform', 'normal',
                      'orthogonal', 'constant', 'ones', 'zeros', 'variance_scaling', 'default']
    units_choices = [32, 64, 128, 256, 512, 1024]
    
    # Agent-specific
    if algorithm in ['DDPG', 'TD3', 'SAC']:
        add_param(param_space, 'actor_num_layers', choices=[1, 2, 3])
        add_param(param_space, 'actor_layer_type', choices=layer_choices)
        add_param(param_space, 'actor_layer_units', choices=units_choices)
        add_param(param_space, 'critic_num_layers', choices=[1, 2, 3])
        add_param(param_space, 'critic_layer_type', choices=layer_choices)
        add_param(param_space, 'critic_layer_units', choices=units_choices)
        add_param(param_space, 'value_num_layers', choices=[1, 2, 3])
        add_param(param_space, 'value_layer_type', choices=layer_choices)
        add_param(param_space, 'value_layer_units', choices=units_choices)
        add_param(param_space, 'actor_output_layer_type', choices=layer_choices)
        add_param(param_space, 'critic_output_layer_type', choices=layer_choices)
        add_param(param_space, 'value_output_layer_type', choices=layer_choices)
        add_param(param_space, 'actor_output_layer_units', choices=units_choices)
        add_param(param_space, 'critic_output_layer_units', choices=units_choices)
        add_param(param_space, 'value_output_layer_units', choices=units_choices)
        
        add_param(param_space, 'tau', default_min=0.001, default_max=0.01)
        add_param(param_space, 'warmup', default_min=500, default_max=2000, is_int=True)
        add_param(param_space, 'N', choices=[1, 2, 3])
        add_param(param_space, 'batch_size', choices=[32, 64, 128, 256])
        add_param(param_space, 'buffer_size', default_min=1e4, default_max=1e6, is_log=True)
        add_param(param_space, 'per_alpha', default_min=0.4, default_max=0.7)
        add_param(param_space, 'per_beta', default_min=0.3, default_max=1.0)
            if algorithm == 'DDPG':
            add_param(param_space, 'action_epsilon', default_min=0.1, default_max=0.3)
        elif algorithm == 'TD3':
            add_param(param_space, 'actor_update_delay', choices=[1, 2, 3])
            add_param(param_space, 'target_noise_clip', default_min=0.3, default_max=0.7)
        elif algorithm == 'SAC':
            add_param(param_space, 'alpha', default_min=0.1, default_max=1.0, is_log=True)
            add_param(param_space, 'auto_entropy_tuning', choices=[True, False])
            add_param(param_space, 'alpha_lr', default_min=1e-4, default_max=1e-3, is_log=True)
    elif algorithm == 'PPO':
        add_param(param_space, 'gae_coefficient', default_min=0.9, default_max=1.0)
        add_param(param_space, 'policy_clip', default_min=0.1, default_max=0.3)
        add_param(param_space, 'value_clip', default_min=0.1, default_max=0.3)
        add_param(param_space, 'value_loss_coefficient', default_min=0.5, default_max=1.0)
        add_param(param_space, 'entropy_coefficient', default_min=0.001, default_max=0.1, is_log=True)
        add_param(param_space, 'normalize_advantages', choices=[True, False])
        add_param(param_space, 'grad_clip', default_min=0.5, default_max=2.0)
    elif algorithm == 'ActorCritic':
        add_param(param_space, 'policy_trace_decay', default_min=0.0, default_max=0.5)
        add_param(param_space, 'value_trace_decay', default_min=0.0, default_max=0.5)
    elif algorithm == 'HER':
        add_param(param_space, 'strategy', choices=['final', 'random', 'future'])
        add_param(param_space, 'tolerance', default_min=0.1, default_max=1.0)
        add_param(param_space, 'num_goals', choices=[2, 4, 8])
        add_param(param_space, 'base_algorithm', choices=['DDPG', 'TD3', 'SAC'])
    
    # Rest of the function remains similar
    searcher = {
        'optuna': OptunaSearch(),
        'hyperopt': HyperOptSearch()
    }.get(user_config['searcher'], None)
    
    scheduler = ASHAScheduler(
        max_t=user_config['max_iterations'],
        grace_period=user_config.get('grace_period', 10),
        reduction_factor=3
    )
    
    resources_per_trial = {
        'cpu': user_config.get('cpus_per_trial', 4),
        'gpu': user_config.get('gpus_per_trial', 0)
    }
    
    tuner = Tuner(
        tune.with_resources(rl_trainable, resources_per_trial),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric='mean_reward',
            mode='max',
            searcher=searcher,
            scheduler=scheduler,
            num_samples=user_config['num_samples'],
            max_concurrent_trials=user_config.get('max_concurrent', 4)
        ),
        run_config=ray.train.RunConfig(
            name='phoenx_rl_sweep',
            stop={'training_iteration': user_config['max_iterations']},
            verbose=1,
            callbacks=[WandbLoggerCallback(
                project=user_config.get('wandb_project', 'phoenx-rl'),
                log_config=True,
                upload_checkpoints=True
            )]
        )
    )
    
    results = tuner.fit()
    best_config = results.get_best_result().config
    return best_config, results