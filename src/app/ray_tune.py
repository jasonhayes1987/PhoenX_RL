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
            {"type": "NStepReward", "params": {"n": config['N']}}
        ]
        env_wrap = GymnasiumWrapper(env_spec, wrappers)
        # env = SyncVectorEnv([lambda: GymnasiumWrapper(env_spec, wrappers) for _ in range(config.get('num_envs', 1))])

        # Component instantiation
        if 'replay_buffer_type' in config:
            buffer_params = {k.replace('replay_buffer_', ''): config[k] for k in config if k.startswith('replay_buffer_')}
            # buffer_params['device'] = config.get('replay_buffer_device', 'cpu')
            buffer = Buffer.create_instance(config['replay_buffer_type'], env=env_wrap, **buffer_params)

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
    """Runs ray tune sweep

    Args:
        user_config (dict): User config for ray tune sweep

    Returns:
        dict: Formatted config for ray tune sweep (with samplers set up, no sampling done)
    """
    ray.init(ignore_reinit_error=True)
    params = user_config['param_space']  # params to be formatted
    param_space = {
        'env_name': user_config['env'],
        'algorithm': user_config['algorithm'],
        'wandb_project': user_config['wandb_project'],
        'max_iterations': user_config['max_iterations'],
        'save_dir': user_config.get('save_dir', 'Tune_Results'),
    }

    def add_param(name, param_type, value):
        if param_type == "choice":
            param_space[name] = tune.choice(value)
        elif param_type == "log":
            param_space[name] = tune.loguniform(value[0], value[1])
        elif param_type == "int":
            param_space[name] = tune.randint(value[0], value[1])
        elif param_type == "uniform":
            param_space[name] = tune.uniform(value[0], value[1])
        elif param_type == "conditional":
            # value is the dict with 'depends_on' and 'conditions'
            depends_on = value['depends_on']
            conditions = value['conditions']
            
            def conditional_sampler(spec):
                parent_value = spec.config.get(depends_on)  # Get the sampled parent value
                sub_params = conditions.get(parent_value, {})  # Get matching sub-dict or empty
                #DEBUG
                print(f'conditional_sampler: depends_on: {depends_on}')
                print(f'conditional_sampler: parent_value: {parent_value}')
                print(f'conditional_sampler: sub_params: {sub_params}')
                result = {}
                for sub_key, sub_def in sub_params.items():
                    sub_type = list(sub_def.keys())[0]  # e.g., "uniform"
                    sub_value = sub_def[sub_type]
                    if sub_type == "choice":
                        result[sub_key] = tune.choice(sub_value).sample()
                    elif sub_type == "uniform":
                        result[sub_key] = tune.uniform(sub_value[0], sub_value[1]).sample()
                    elif sub_type == "log":
                        result[sub_key] = tune.loguniform(sub_value[0], sub_value[1]).sample()
                    elif sub_type == "int":
                        result[sub_key] = tune.randint(sub_value[0], sub_value[1]).sample()
                    elif sub_type == "default":
                        result[sub_key] = sub_value  # Fixed value
                    else:
                        result[sub_key] = None  # Or raise error if unsupported
                return result
            
            param_space[name] = tune.sample_from(conditional_sampler)

    def _flatten_config(current_dict, current_prefix=""):
        """Recursive helper to flatten nested dict and build prefixed keys."""
        for key, value in current_dict.items():
            full_key = f"{current_prefix}{key}" if current_prefix else key
            
            if isinstance(value, dict):
                if "conditional" in value:  # Existing: Handle explicit conditionals
                    add_param(full_key, "conditional", value["conditional"])
                    # Process non-conditional parts
                    non_conditional = {k: v for k, v in value.items() if k != "conditional"}
                    _flatten_config(non_conditional, current_prefix=f"{full_key}_")
                elif key.endswith("layer_config"):  # New: Special handling for layer configs
                    # Assume depends_on is sibling "num_layers" (e.g., "actor_model_num_layers")
                    stem = key[:-len("layer_config")]  # "" | "state_" | "merged_"
                    if stem:
                        depends_on = f"{current_prefix}{stem}num_layers"
                    else:
                        depends_on = f"{current_prefix}num_layers"

                    def layer_sampler(spec):
                        num_layers = spec.config.get(depends_on, 0)
                        result = {}
                        for layer_num_str, layer_def in value.items():
                            layer_num = int(layer_num_str)
                            if num_layers >= layer_num:
                                layer_result = {}
                                if isinstance(layer_def, dict):
                                    dependent_values = {}
                                    for sub_k, sub_v in layer_def.items():
                                        if sub_k == "conditional":
                                            continue
                                        sub_type = list(sub_v.keys())[0]
                                        sub_val = sub_v[sub_type]
                                        if sub_type == "choice":
                                            sampled_value = tune.choice(sub_val).sample()
                                            layer_result[sub_k] = sampled_value
                                            dependent_values[sub_k] = sampled_value
                                        elif sub_type == "uniform":
                                            sampled_value = tune.uniform(sub_val[0], sub_val[1]).sample()
                                            layer_result[sub_k] = sampled_value
                                            dependent_values[sub_k] = sampled_value
                                        elif sub_type == "log":
                                            # BUGFIX: use sub_val for both bounds
                                            sampled_value = tune.loguniform(sub_val[0], sub_val[1]).sample()
                                            layer_result[sub_k] = sampled_value
                                            dependent_values[sub_k] = sampled_value
                                        elif sub_type == "int":
                                            sampled_value = tune.randint(sub_val[0], sub_val[1]).sample()
                                            layer_result[sub_k] = sampled_value
                                            dependent_values[sub_k] = sampled_value
                                        elif sub_type == "default":
                                            layer_result[sub_k] = sub_val
                                            dependent_values[sub_k] = sub_val

                                    if "conditional" in layer_def:
                                        cond_value = layer_def["conditional"]
                                        cond_depends_on = cond_value["depends_on"]  # e.g., "actor_model_layer_config_1_type"
                                        dependent_key = cond_depends_on.split("_")[-1]  # "type"
                                        parent_value = dependent_values.get(dependent_key)
                                        sub_params = cond_value["conditions"].get(parent_value, {})
                                        for sub_key, sub_def in sub_params.items():
                                            sub_type = list(sub_def.keys())[0]
                                            sub_val = sub_def[sub_type]
                                            if sub_type == "choice":
                                                layer_result[sub_key] = tune.choice(sub_val).sample()
                                            elif sub_type == "uniform":
                                                layer_result[sub_key] = tune.uniform(sub_val[0], sub_val[1]).sample()
                                            elif sub_type == "log":
                                                layer_result[sub_key] = tune.loguniform(sub_val[0], sub_val[1]).sample()
                                            elif sub_type == "int":
                                                layer_result[sub_key] = tune.randint(sub_val[0], sub_val[1]).sample()
                                            elif sub_type == "default":
                                                layer_result[sub_key] = sub_val

                                result[layer_num_str] = layer_result
                        return result

                    param_space[full_key] = tune.sample_from(layer_sampler)
                elif list(value.keys())[0] in ["choice", "uniform", "log", "int", "default"]:
                    # Leaf node
                    param_type = list(value.keys())[0]
                    param_value = value[param_type]
                    add_param(full_key, param_type, param_value)
                else:
                    # Recurse into sub-dict
                    _flatten_config(value, current_prefix=f"{full_key}_")
            else:
                # Direct value (assume default)
                add_param(full_key, "default", value)

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
            search_alg=searcher,
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