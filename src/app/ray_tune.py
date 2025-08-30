import ray
from ray import tune, air
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air.integrations.wandb import WandbLoggerCallback
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from agent_utils import get_agent_class_from_type
import torch as T
from models import ActorModel, CriticModel, ValueModel, select_policy_model
from buffer import Buffer
from noise import Noise
from normalizer import Normalizer
from icm import ICM
from env_wrapper import GymnasiumWrapper
from schedulers import ScheduleWrapper

def build_layer_config(config, prefix, max_layers=6):
    num_layers = config[f'{prefix}num_layers']
    layer_config = []
    for i in range(1, num_layers + 1):
        type = config[f'{prefix}layer{i}_type']
        params = {}
        if type in ['dense', 'conv']:
            params['kernel'] = config[f'{prefix}layer{i}_init']
            params['kernel params'] = config.get(f'{prefix}layer{i}_kernel_params', {})
            if type == 'dense':
                params['units'] = config[f'{prefix}layer{i}_units']
            if type == 'conv':
                params['out_channels'] = config[f'{prefix}layer{i}_out_channels']
                params['kernel_size'] = config[f'{prefix}layer{i}_kernel_size']
                params['stride'] = config[f'{prefix}layer{i}_stride']
                params['padding'] = config[f'{prefix}layer{i}_padding']
        elif type == 'dropout':
            params['rate'] = config[f'{prefix}layer{i}_rate']
        elif type == 'pool':
            params['pool_type'] = config[f'{prefix}layer{i}_pool_type']
            params['kernel_size'] = config[f'{prefix}layer{i}_kernel_size']
            params['stride'] = config[f'{prefix}layer{i}_stride']
        elif type == 'batchnorm1d':
            params = {}  # No specific params needed
        # Add for other types (e.g., relu, tanh need no params)
        layer_config.append({'type': type, 'params': params})
    return layer_config

def rl_trainable(config):
    try:
        # Create environment with wrappers
        env = gym.make(config['env_name'])
        env_spec = env.spec
        wrappers = [
            {"type": "NStepReward", "params": {"n": config['n_step_n']}}
        ]
        env_wrap = GymnasiumWrapper(env_spec, wrappers)
        # env = SyncVectorEnv([lambda: GymnasiumWrapper(env_spec, wrappers) for _ in range(config.get('num_envs', 1))])

        # Component instantiation
        if 'buffer_type' in config:
            buffer_params = {k.replace('buffer_', ''): config[k] for k in config if k.startswith('buffer_')}
            buffer_params['device'] = config.get('buffer_device', 'cpu')
            buffer = Buffer.create_instance(config['buffer_type'], env=env_wrap, **buffer_params)

        if 'noise_type' in config:
            noise_params = {k.replace('noise_', ''): config[k] for k in config if k.startswith('noise_')}
            noise_params['device'] = config.get('device', 'cuda')
            noise = Noise.create_instance(config['noise_type'], **noise_params)

        normalizer_params = {k.replace('normalizer_', ''): config[k] for k in config if k.startswith('normalizer_')}
        normalizer_params['size'] = env_wrap.single_observation_space.shape[0]
        normalizer_params['device'] = config.get('buffer_device', 'cpu')
        normalizer = Normalizer.create_instance(config['normalizer_type'], **normalizer_params)

        icm = None
        if config.get('use_icm', False):
            encoder_config = build_layer_config(config, 'icm_encoder_')
            inverse_config = build_layer_config(config, 'icm_inverse_')
            forward_config = build_layer_config(config, 'icm_forward_')
            model_configs = {
                'encoder': {'layer_config': encoder_config, 'output_layer': {'type': 'dense', 'params': {'kernel': config['icm_encoder_output_layer_init'], 'kernel params': {}}}},
                'inverse_model': {'layer_config': inverse_config, 'output_layer': {'type': 'dense', 'params': {'kernel': config['icm_inverse_output_layer_init'], 'kernel params': {}}}},
                'forward_model': {'layer_config': forward_config, 'output_layer': {'type': 'dense', 'params': {'kernel': config['icm_forward_output_layer_init'], 'kernel params': {}}}},
            }
            icm_params = {k.replace('icm_', ''): config[k] for k in config if k.startswith('icm_') and not k.startswith(('icm_encoder_', 'icm_inverse_', 'icm_forward_'))}
            icm_params['model_configs'] = model_configs
            icm_params['reward_scheduler'] = ScheduleWrapper({
                'type': config['icm_scheduler_type'],
                'params': {
                    'start_factor': config['icm_scheduler_start_factor'],
                    'end_factor': config['icm_scheduler_end_factor'],
                    'total_iters': config['icm_scheduler_total_iters']
                }
            })
            icm_params['device'] = config.get('device', 'cuda')
            icm = ICM.create_instance(env=env_wrap, **icm_params)

        # Build models
        obs_shape = env.single_observation_space.shape
        action_dim = env.single_action_space.shape[0] if isinstance(env.single_action_space, gym.spaces.Box) else env.single_action_space.n
        continuous = isinstance(env.single_action_space, gym.spaces.Box)

        actor_layer_config = build_layer_config(config, 'actor_')
        actor_optimizer_params = {'type': config['actor_optimizer_type'], 'params': {'lr': config['actor_lr']}}
        output_layer_config = [{'type': 'dense', 'params': {'kernel': config['actor_output_layer_init'], 'kernel params': {}}}]
        actor_model = ActorModel(env_wrap, layer_config=actor_layer_config, output_layer_config=output_layer_config, optimizer_params=actor_optimizer_params, device=config.get('device', 'cuda'))

        # Critic uses actor's layer_config as merged_layers, empty state_layers
        critic_optimizer_params = {'type': config['critic_optimizer_type'], 'params': {'lr': config['critic_lr']}}
        critic_model_a = CriticModel(env_wrap, state_layers=[], merged_layers=actor_layer_config, output_layer_config=output_layer_config, optimizer_params=critic_optimizer_params, device=config.get('device', 'cuda'))
        critic_model_b = critic_model_a.clone()

        value_layer_config = build_layer_config(config, 'value_')
        value_optimizer_params = {'type': config['value_optimizer_type'], 'params': {'lr': config['value_lr']}}
        value_model = ValueModel(env_wrap, layer_config=value_layer_config, output_layer_config=output_layer_config, optimizer_params=value_optimizer_params, device=config.get('device', 'cuda'))

        # Agent
        agent_class = get_agent_class_from_type('SAC')
        agent_kwargs = {
            'env': env,
            'actor_model': actor_model,
            'critic_model_a': critic_model_a,
            'critic_model_b': critic_model_b,
            'value_model': value_model,
            'replay_buffer': buffer,
            'discount': config['gamma'],
            'tau': config['tau'],
            'alpha': config['alpha'],
            'auto_entropy_tuning': config['auto_entropy_tuning'],
            'alpha_lr': config['alpha_lr'],
            'batch_size': config['batch_size'],
            'grad_clip': config['grad_clip'] if config['grad_clip'] != 'inf' else float('inf'),
            'warmup': config['warmup'],
            'N': config['buffer_N'],
            'curiosity': icm,
            'state_normalizer': normalizer,
            'callbacks': [WandbCallback(project_name='HumanoidStandup-v5')],
            'save_dir': config.get('save_dir', 'HumanoidStandup_N3'),
            'device': config.get('device', 'cuda')
        }

        agent = agent_class(**agent_kwargs)

        for iteration in range(config['max_iterations']):
            metrics = agent.train_step()
            session.report(metrics)

        agent.save(config['save_dir'])
        return metrics
    except ValueError as e:
        print(f"Invalid config: {e}")
        session.report({'mean_reward': -float('inf')})
        return {'mean_reward': -float('inf')}

def run_ray_tune_sweep(user_config):
    ray.init(ignore_reinit_error=True)

    """Formats user config for ray tune sweep

    Args:
        user_config (dict): User config for ray tune sweep

    Returns:
        dict: Formatted config for ray tune sweep
    """
    params = user_config['param_space']  # params to be formatted
    param_space = {
        'env_name': user_config['env'],
        'algorithm': user_config['algorithm'],
        'wandb_project': user_config['wandb_project'],
        'max_iterations': user_config['max_iterations'],
        'save_dir': user_config.get('save_dir', 'Tune_Results'),
    }

    def add_param(space, name, param_type, value):
        if param_type == "choice":
            space[name] = tune.choice(value)
        elif param_type == "log":
            space[name] = tune.loguniform(value[0], value[1])
        elif param_type == "int":
            space[name] = tune.randint(value[0], value[1])
        elif param_type == "uniform":
            space[name] = tune.uniform(value[0], value[1])

    def _flatten_config(current_dict, current_prefix=""):
        """Recursive helper to flatten nested dict and build prefixed keys."""
        for key, value in current_dict.items():
            # Build the full key with prefix (use '_' as separator)
            full_key = f"{current_prefix}{key}" if current_prefix else key
            
            if isinstance(value, dict):
                # Recurse deeper, appending current key to prefix
                if list(value.keys())[0] in ["choice", "uniform", "log", "int", "default"]:
                    # It's a leaf: extract param_type and value, then add to space
                    param_type = list(value.keys())[0]
                    param_value = value[param_type]
                    add_param(param_space, full_key, param_type, param_value)
                else:
                    # Not a leaf: recurse with updated prefix
                    _flatten_config(value, current_prefix=f"{full_key}_")
            else:
                # Handle non-dict (though your structure seems to always have dicts at leaves)
                # If needed, adapt this for direct values (e.g., assume "default" type)
                add_param(param_space, full_key, "default", value)

    # Start the recursion from the top-level params
    _flatten_config(params)
    searcher = {
        'optuna': OptunaSearch(),
        'hyperopt': HyperOptSearch()
    }.get(user_config['searcher'], None)

    scheduler = ASHAScheduler(
        max_t=user_config['max_iterations'],
        grace_period=user_config.get('grace_period', 10000),
        reduction_factor=3
    )

    resources_per_trial = {'cpu': user_config.get('cpus_per_trial', 4), 'gpu': user_config.get('gpus_per_trial', 1)}

    tuner = tune.Tuner(
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
        run_config=air.RunConfig(
            name='phoenx_rl_sweep',
            stop={'training_iteration': user_config['max_iterations']},
            verbose=1,
            callbacks=[WandbLoggerCallback(project=user_config['wandb_project'], log_config=True, upload_checkpoints=True)]
        )
    )

    results = tuner.fit()
    best_config = results.get_best_result().config
    return best_config, results