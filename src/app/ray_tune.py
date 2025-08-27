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
            params['units'] = config[f'{prefix}layer{i}_units']
            params['kernel'] = config[f'{prefix}layer{i}_init']
            params['kernel params'] = config.get(f'{prefix}layer{i}_kernel_params', {})
            if type == 'conv':
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
            {"type": "NStepReward", "params": {"n": config['n_step_n'], "discount": config['gamma']}}
        ]
        env_wrap = GymnasiumWrapper(env_spec, wrappers)
        env = SyncVectorEnv([lambda: GymnasiumWrapper(env_spec, wrappers) for _ in range(config.get('num_envs', 1))])

        # Component instantiation
        buffer_params = {k.replace('buffer_', ''): config[k] for k in config if k.startswith('buffer_')}
        buffer_params['device'] = config.get('buffer_device', 'cpu')
        buffer = Buffer.create_instance(config['buffer_type'], env=env_wrap, **buffer_params)

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

    algorithm = user_config['algorithm']

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
            space[name] = user_config[name]
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
        'wandb_project': user_config.get('wandb_project', 'phoenx-rl'),
        'max_iterations': user_config['max_iterations'],
        'save_dir': user_config.get('save_dir', 'HumanoidStandup_N3'),
        'device': user_config.get('device', 'cuda'),
        'buffer_device': user_config.get('buffer_device', 'cpu'),
    }

    # Common
    add_param(param_space, 'gamma', choices=user_config.get('gamma_choices', [0.99, 0.98]))
    add_param(param_space, 'batch_size', choices=user_config.get('batch_size_choices', [256, 512, 1024]))
    add_param(param_space, 'n_step_n', choices=user_config.get('n_step_n_choices', [1, 3, 5]))

    # Buffer
    add_param(param_space, 'buffer_type', choices=['ReplayBuffer'])
    add_param(param_space, 'buffer_size', default=user_config.get('buffer_size', 1000000))
    add_param(param_space, 'buffer_N', choices=user_config.get('buffer_N_choices', [1, 3, 5]))

    # Noise
    add_param(param_space, 'noise_type', choices=['NormalNoise', 'UniformNoise', 'OUNoise'])
    add_param(param_space, 'noise_mean', default=0.0)
    add_param(param_space, 'noise_stddev', default_min=0.1, default_max=0.5)
    add_param(param_space, 'noise_minval', default_min=-1.0, default_max=0.0)
    add_param(param_space, 'noise_maxval', default_min=0.0, default_max=1.0)
    add_param(param_space, 'noise_theta', default_min=0.1, default_max=0.2)
    add_param(param_space, 'noise_sigma', default_min=0.1, default_max=0.3)
    add_param(param_space, 'noise_dt', default_min=1e-3, default_max=1e-1, is_log=True)

    # Normalizer
    add_param(param_space, 'normalizer_type', choices=['Normalizer'])
    add_param(param_space, 'normalizer_clip_range', choices=user_config.get('normalizer_clip_range_choices', [1.0, 5.0, 10.0]))
    add_param(param_space, 'normalizer_eps', default_min=1e-8, default_max=1e-4, is_log=True)

    # ICM
    add_param(param_space, 'use_icm', choices=[True])
    add_param(param_space, 'icm_reward_weight', default=user_config.get('icm_reward_weight', 0.2))
    add_param(param_space, 'icm_beta', default=user_config.get('icm_beta', 0.2))
    add_param(param_space, 'icm_extrinsic_threshold', default=user_config.get('icm_extrinsic_threshold', 1000))
    add_param(param_space, 'icm_warmup', default=user_config.get('icm_warmup', 1000))
    for sub in ['encoder', 'inverse_model', 'forward_model']:
        add_param(param_space, f'icm_{sub}_num_layers', choices=user_config[f'icm_{sub}_num_layers_choices'])
        for i in range(1, 7):
            add_param(param_space, f'icm_{sub}_layer{i}_type', choices=user_config[f'icm_{sub}_layer{i}_type_choices'])
            add_param(param_space, f'icm_{sub}_layer{i}_units', choices=user_config[f'icm_{sub}_layer{i}_units_choices'])
            add_param(param_space, f'icm_{sub}_layer{i}_init', choices=user_config[f'icm_{sub}_layer{i}_init_choices'])
        add_param(param_space, f'icm_{sub}_output_layer_init', choices=user_config[f'icm_{sub}_output_layer_init_choices'])
    add_param(param_space, 'icm_scheduler_type', choices=user_config['icm_scheduler_type_choices'])
    add_param(param_space, 'icm_scheduler_start_factor', choices=user_config['icm_scheduler_start_factor_choices'])
    add_param(param_space, 'icm_scheduler_end_factor', choices=user_config['icm_scheduler_end_factor_choices'])
    add_param(param_space, 'icm_scheduler_total_iters', choices=user_config['icm_scheduler_total_iters_choices'])
    add_param(param_space, 'icm_optimizer_type', choices=['Adam'])
    add_param(param_space, 'icm_lr', default_min=user_config['icm_lr_min'], default_max=user_config['icm_lr_max'], is_log=True)

    # Models
    layer_choices = ['dense', 'batchnorm1d', 'relu']
    weight_choices = ['default', 'XavierUniform', 'KaimingUniform']
    units_choices = [128, 256, 512]
    max_layers = 6

    for prefix in ['actor', 'value']:
        add_param(param_space, f'{prefix}_num_layers', choices=user_config[f'{prefix}_num_layers_choices'])
        for i in range(1, max_layers + 1):
            add_param(param_space, f'{prefix}_layer{i}_type', choices=user_config[f'{prefix}_layer{i}_type_choices'])
            # Customize units for layer 3
            if i == 3:
                add_param(param_space, f'{prefix}_layer{i}_units', choices=[128, 256])
            else:
                add_param(param_space, f'{prefix}_layer{i}_units', choices=user_config[f'{prefix}_layer{i}_units_choices'])
            add_param(param_space, f'{prefix}_layer{i}_init', choices=user_config[f'{prefix}_layer{i}_init_choices'])
            add_param(param_space, f'{prefix}_layer{i}_kernel_params', default={})
        add_param(param_space, f'{prefix}_output_layer_init', choices=user_config[f'{prefix}_output_layer_init_choices'])
        add_param(param_space, f'{prefix}_optimizer_type', choices=[user_config[f'{prefix}_optimizer_type']])
        add_param(param_space, f'{prefix}_lr', default_min=user_config[f'{prefix}_lr_min'], default_max=user_config[f'{prefix}_lr_max'], is_log=True)

    # Critic uses actor's merged_layers
    add_param(param_space, 'critic_optimizer_type', choices=[user_config['critic_optimizer_type']])
    add_param(param_space, 'critic_lr', default_min=user_config['critic_lr_min'], default_max=user_config['critic_lr_max'], is_log=True)

    # SAC-specific
    add_param(param_space, 'tau', choices=user_config['tau_choices'])
    add_param(param_space, 'alpha', choices=user_config['alpha_choices'])
    add_param(param_space, 'auto_entropy_tuning', choices=user_config['auto_entropy_tuning_choices'])
    add_param(param_space, 'alpha_lr', default_min=user_config['alpha_lr_min'], default_max=user_config['alpha_lr_max'], is_log=True)
    add_param(param_space, 'grad_clip', choices=user_config['grad_clip_choices'])
    add_param(param_space, 'warmup', choices=user_config['warmup_choices'])

    searcher = {
        'optuna': OptunaSearch(),
        'hyperopt': HyperOptSearch()
    }.get(user_config['searcher'], None)

    scheduler = ASHAScheduler(
        max_t=user_config['max_iterations'],
        grace_period=user_config.get('grace_period', 1000),
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