import ray
from ray import tune, air
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air.integrations.wandb import WandbLoggerCallback
from rl_callbacks import WandbCallback as RLWandbCallback
import inspect
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
        # Helpers to parse flattened config produced by _flatten_config
        def _collect_kernel_params(cfg, base_prefix: str):
            params = {}
            # Accept both "<base>_kernel_params_<suffix>" and "<base>_<suffix>"
            for suffix in ["mode", "gain", "scale", "distribution"]:
                key1 = f"{base_prefix}_kernel_params_{suffix}"
                key2 = f"{base_prefix}_{suffix}"
                if key1 in cfg:
                    params[suffix] = cfg[key1]
                if key2 in cfg:
                    params[suffix] = cfg[key2]
            return params

        def _sanitize_kernel_params(kernel_name: str | None, params: dict) -> dict:
            if not kernel_name or not isinstance(params, dict):
                return params or {}
            k = str(kernel_name).lower()
            allowed = {}
            if 'kaiming' in k:
                for name in ['mode', 'nonlinearity', 'a']:
                    if name in params:
                        allowed[name] = params[name]
                return allowed
            if 'xavier' in k:
                if 'gain' in params:
                    allowed['gain'] = params['gain']
                return allowed
            if k == 'variance_scaling':
                for name in ['scale', 'mode', 'distribution']:
                    if name in params:
                        allowed[name] = params[name]
                return allowed
            return params

        def _build_layer_list(layer_dict: dict, kernel: str | None, kernel_params: dict):
            """Convert dict of {'1': {...}, '2': {...}} → ordered list expected by models."""
            if not isinstance(layer_dict, dict):
                return []

            layer_list = []
            for idx in sorted(layer_dict.keys(), key=lambda x: int(x)):
                spec = layer_dict[idx] if isinstance(idx, int) else layer_dict[str(idx)]
                if not isinstance(spec, dict):
                    continue
                layer_type = spec.get("type")
                params = {}
                if layer_type == "dense":
                    if "units" in spec:
                        params["units"] = spec["units"]
                elif layer_type == "conv":
                    for k in ["out_channels", "kernel_size", "stride", "padding"]:
                        if k in spec:
                            params[k] = spec[k]
                elif layer_type == "dropout":
                    if "rate" in spec:
                        params["rate"] = spec["rate"]
                elif layer_type == "pool":
                    for k in ["pool_type", "kernel_size", "stride"]:
                        if k in spec:
                            params[k] = spec[k]
                # Add initializer if applicable
                if layer_type in ["dense", "conv"] and kernel is not None:
                    params["kernel"] = kernel
                    params["kernel params"] = _sanitize_kernel_params(kernel, kernel_params)
                layer_list.append({"type": layer_type, "params": params})
            return layer_list

        def _build_scheduler(cfg, prefix: str, suffix: str):
            # suffix in {'lr_scheduler','reward_scheduler'}
            type_key = f"{prefix}_{suffix}_type"
            sched_type = cfg.get(type_key, None)
            if not sched_type:
                return None
            stype = str(sched_type).lower()
            params = {}
            def get(name: str):
                return cfg.get(f"{prefix}_{suffix}_{name}")
            if stype == 'linear':
                for k in ['start_factor', 'end_factor', 'total_iters']:
                    v = get(k)
                    if v is not None:
                        params[k] = v
            elif stype == 'step':
                if get('step_size') is not None:
                    params['step_size'] = get('step_size')
                gamma = get('gamma') or get('step_gamma')
                if gamma is not None:
                    params['gamma'] = gamma
                # Ensure required step_size exists
                if 'step_size' not in params:
                    # Fallback to a reasonable default if not provided
                    params['step_size'] = int(cfg.get('max_iterations', 10000))
            elif stype == 'cosineannealing':
                total_iters = get('total_iters')
                params['T_max'] = total_iters if total_iters is not None else 100000
                if get('min_lr') is not None:
                    params['eta_min'] = get('min_lr')
            elif stype == 'exponential':
                gamma = get('gamma') or get('exponential_gamma')
                if gamma is None:
                    return None
                params['gamma'] = gamma
            else:
                return None
            return ScheduleWrapper({"type": stype, "params": params})

        def _build_model_scheduler(cfg, base_prefix: str):
            return _build_scheduler(cfg, base_prefix, 'lr_scheduler')

        # Create environment with wrappers
        env = gym.make(config['env_name'])
        env_spec = env.spec
        wrappers = [
            {"type": "NStepReward", "params": {"n": config.get('N', 1)}}
        ]
        env_wrap = GymnasiumWrapper(env_spec, wrappers)
        # env = SyncVectorEnv([lambda: GymnasiumWrapper(env_spec, wrappers) for _ in range(config.get('num_envs', 1))])

        # Component instantiation
        buffer = None
        if 'replay_buffer_type' in config:
            buffer_type = config['replay_buffer_type']
            # Map flattened keys to Buffer constructor
            buffer_params = {}
            if 'replay_buffer_size' in config:
                buffer_params['buffer_size'] = config['replay_buffer_size']
            if 'replay_buffer_device' in config:
                buffer_params['device'] = config['replay_buffer_device']
            # Optional N-step for buffers
            if 'N' in config:
                buffer_params['N'] = config['N']
            # Prioritized-specific
            key_map = {
                'replay_buffer_alpha': 'alpha',
                'replay_buffer_beta_start': 'beta_start',
                'replay_buffer_beta_iter': 'beta_iter',
                'replay_buffer_beta_update_freq': 'beta_update_freq',
                'replay_buffer_priority_type': 'priority',
                'replay_buffer_normalize': 'normalize',
            }
            for k_flat, k_buf in key_map.items():
                if k_flat in config:
                    buffer_params[k_buf] = config[k_flat]
            buffer = Buffer.create_instance(buffer_type, env=env_wrap, **buffer_params)

        noise = None
        if 'noise_type' in config:
            noise_params = {k.replace('noise_', ''): config[k] for k in config if k.startswith('noise_')}
            noise_params['device'] = config.get('device', 'cpu')
            noise = Noise.create_instance(config['noise_type'], **noise_params)

        # State/Goal normalizers from flattened keys
        state_normalizer = None
        goal_normalizer = None
        if any(k.startswith('state_normalizer_') for k in config.keys()):
            state_normalizer_size = env_wrap.single_observation_space.shape[0]
            state_normalizer = Normalizer.create_instance(
                size=state_normalizer_size,
                clip_range=config.get('state_normalizer_clip_range', 5.0),
                device=config.get('state_normalizer_device', 'cpu')
            )
        if any(k.startswith('goal_normalizer_') for k in config.keys()):
            goal_normalizer_size = env_wrap.single_observation_space.shape[0]
            goal_normalizer = Normalizer.create_instance(
                size=goal_normalizer_size,
                clip_range=config.get('goal_normalizer_clip_range', 5.0),
                device=config.get('goal_normalizer_device', 'cpu')
            )

        # Curiosity (ICM) from flattened keys under curiosity_*
        icm = None
        if config.get('curiosity_use_icm', False):
            # Encoder
            enc_kernel = config.get('curiosity_encoder_kernel', 'default')
            enc_kparams = _collect_kernel_params(config, 'curiosity_encoder')
            enc_layers = _build_layer_list(
                config.get('curiosity_encoder_layer_config', {}), enc_kernel, enc_kparams
            )
            enc_out_kernel = config.get('curiosity_encoder_output_layer_kernel', 'default')
            enc_out_kparams = _sanitize_kernel_params(
                enc_out_kernel, _collect_kernel_params(config, 'curiosity_encoder_output')
            )
            enc_output = [{'type': 'dense', 'params': {'kernel': enc_out_kernel, 'kernel params': enc_out_kparams}}]

            # Inverse
            inv_kernel = config.get('curiosity_inverse_model_kernel', 'default')
            inv_kparams = _collect_kernel_params(config, 'curiosity_inverse_model')
            inv_layers = _build_layer_list(
                config.get('curiosity_inverse_model_layer_config', {}), inv_kernel, inv_kparams
            )
            inv_out_kernel = config.get('curiosity_inverse_model_output_layer_kernel', 'default')
            inv_out_kparams = _sanitize_kernel_params(
                inv_out_kernel, _collect_kernel_params(config, 'curiosity_inverse_model_output')
            )
            inv_output = [{'type': 'dense', 'params': {'kernel': inv_out_kernel, 'kernel params': inv_out_kparams}}]

            # Forward
            fwd_kernel = config.get('curiosity_forward_model_kernel', 'default')
            fwd_kparams = _collect_kernel_params(config, 'curiosity_forward_model')
            fwd_layers = _build_layer_list(
                config.get('curiosity_forward_model_layer_config', {}), fwd_kernel, fwd_kparams
            )
            fwd_out_kernel = config.get('curiosity_forward_model_output_layer_kernel', 'default')
            fwd_out_kparams = _sanitize_kernel_params(
                fwd_out_kernel, _collect_kernel_params(config, 'curiosity_forward_model_output')
            )
            fwd_output = [{'type': 'dense', 'params': {'kernel': fwd_out_kernel, 'kernel params': fwd_out_kparams}}]

            model_configs = {
                'encoder': {'layer_config': enc_layers, 'output_layer': enc_output},
                'inverse_model': {'layer_config': inv_layers, 'output_layer': inv_output},
                'forward_model': {'layer_config': fwd_layers, 'output_layer': fwd_output},
            }
            icm_optimizer = {
                'type': config.get('curiosity_optimizer_params_type', 'adam'),
                'params': {'lr': config.get('curiosity_optimizer_params_lr', 1e-3)}
            }
            reward_scheduler = _build_scheduler(config, 'curiosity_reward', 'scheduler')

            icm = ICM.create_instance(
                env=env_wrap,
                model_configs=model_configs,
                optimizer_params=icm_optimizer,
                reward_weight=config.get('curiosity_reward_weight', 0.1),
                reward_scheduler=reward_scheduler,
                beta=config.get('curiosity_beta', 0.2),
                extrinsic_threshold=config.get('curiosity_extrinsic_threshold', 0),
                warmup=config.get('curiosity_warmup', 0),
                device=config.get('curiosity_device', config.get('device', 'cuda'))
            )

        # Build models
        obs_shape = env_wrap.single_observation_space.shape
        action_dim = env_wrap.single_action_space.shape[0] if isinstance(env_wrap.single_action_space, gym.spaces.Box) else env_wrap.single_action_space.n
        continuous = isinstance(env_wrap.single_action_space, gym.spaces.Box)

        # ActorModel from flattened keys under actor_model_*
        actor_kernel = config.get('actor_model_kernel', 'default')
        actor_kparams = _collect_kernel_params(config, 'actor_model')
        actor_layers = _build_layer_list(config.get('actor_model_layer_config', {}), actor_kernel, actor_kparams)
        actor_out_kernel = config.get('actor_model_output_layer_kernel', 'default')
        actor_out_kparams = _sanitize_kernel_params(
            actor_out_kernel, _collect_kernel_params(config, 'actor_model_output')
        )
        actor_output_layer_kernel = [{'type': 'dense', 'params': {'kernel': actor_out_kernel, 'kernel params': actor_out_kparams}}]
        actor_optimizer_params = {
            'type': config.get('actor_model_optimizer_params_type', 'adam'),
            'params': {'lr': config.get('actor_model_optimizer_params_lr', 1e-3)}
        }
        actor_lr_scheduler = _build_model_scheduler(config, 'actor_model')
        actor_model = ActorModel(
            env_wrap,
            layer_config=actor_layers,
            output_layer_kernel=actor_output_layer_kernel,
            optimizer_params=actor_optimizer_params,
            lr_scheduler=actor_lr_scheduler,
            device=config.get('actor_model_device', config.get('device', 'cuda'))
        )

        # Policy model for on-policy agents (PPO/ActorCritic/Reinforce)
        policy_model = None
        try:
            policy_model_cls = select_policy_model(env)
            # Many policy models accept 'distribution' arg; use actor_model_distribution if present
            distribution = config.get('actor_model_distribution', None)
            policy_kwargs = {
                'env': env_wrap,
                'layer_config': actor_layers,
                'output_layer_kernel': actor_output_layer_kernel,
                'optimizer_params': actor_optimizer_params,
                'lr_scheduler': actor_lr_scheduler,
            }
            if distribution is not None:
                policy_kwargs['distribution'] = distribution
            # Filter kwargs by constructor signature to avoid unexpected params
            pm_sig = inspect.signature(policy_model_cls.__init__)
            allowed_pm = set(pm_sig.parameters.keys()) - {'self'}
            policy_kwargs = {k: v for k, v in policy_kwargs.items() if k in allowed_pm}
            policy_model = policy_model_cls(**policy_kwargs)
        except Exception:
            policy_model = None

        # CriticModel from flattened keys under critic_model_*
        critic_state_layers = []  # optional
        state_num = int(config.get('critic_model_state_num_layers', 0) or 0)
        if state_num > 0 and isinstance(config.get('critic_model_state_layer_config'), dict):
            state_kernel = config.get('critic_model_state_kernel', 'default')
            state_kparams = _collect_kernel_params(config, 'critic_model_state')
            critic_state_layers = _build_layer_list(config.get('critic_model_state_layer_config', {}), state_kernel, state_kparams)

        merged_kernel = config.get('critic_model_merged_kernel', 'default')
        merged_kparams = _collect_kernel_params(config, 'critic_model_merged')
        merged_layers = _build_layer_list(config.get('critic_model_merged_layer_config', {}), merged_kernel, merged_kparams)
        critic_out_kernel = config.get('critic_model_output_layer_kernel', 'default')
        critic_out_kparams = _sanitize_kernel_params(
            critic_out_kernel, _collect_kernel_params(config, 'critic_model_output')
        )
        critic_output_layer_kernel = [{'type': 'dense', 'params': {'kernel': critic_out_kernel, 'kernel params': critic_out_kparams}}]
        critic_optimizer_params = {
            'type': config.get('critic_model_optimizer_params_type', 'adam'),
            'params': {'lr': config.get('critic_model_optimizer_params_lr', 1e-3)}
        }
        critic_lr_scheduler = _build_model_scheduler(config, 'critic_model')
        critic_model_a = CriticModel(
            env_wrap,
            state_layers=critic_state_layers,
            merged_layers=merged_layers,
            output_layer_kernel=critic_output_layer_kernel,
            optimizer_params=critic_optimizer_params,
            lr_scheduler=critic_lr_scheduler,
            device=config.get('critic_model_device', config.get('device', 'cuda'))
        )
        critic_model_b = critic_model_a.clone()

        # ValueModel from flattened keys under value_model_*
        value_kernel = config.get('value_model_kernel', 'default')
        value_kparams = _collect_kernel_params(config, 'value_model')
        value_layers = _build_layer_list(config.get('value_model_layer_config', {}), value_kernel, value_kparams)
        value_out_kernel = config.get('value_model_output_layer_kernel', 'default')
        value_out_kparams = _sanitize_kernel_params(
            value_out_kernel, _collect_kernel_params(config, 'value_model_output')
        )
        value_output_layer_kernel = [{'type': 'dense', 'params': {'kernel': value_out_kernel, 'kernel params': value_out_kparams}}]
        value_optimizer_params = {
            'type': config.get('value_model_optimizer_params_type', 'adam'),
            'params': {'lr': config.get('value_model_optimizer_params_lr', 1e-3)}
        }
        value_lr_scheduler = _build_model_scheduler(config, 'value_model')
        value_model = ValueModel(
            env_wrap,
            layer_config=value_layers,
            output_layer_kernel=value_output_layer_kernel,
            optimizer_params=value_optimizer_params,
            lr_scheduler=value_lr_scheduler,
            device=config.get('value_model_device', config.get('device', 'cuda'))
        )

        # Agent: dynamically assemble kwargs based on constructor signature
        agent_class = get_agent_class_from_type(config.get('algorithm'))
        # Superset of possible kwargs
        common_kwargs = {
            'env': env_wrap,
            'policy_model': policy_model,
            'actor_model': actor_model,
            'critic_model': critic_model_a,  # alias for algorithms expecting single critic
            'critic_model_a': critic_model_a,
            'critic_model_b': critic_model_b,
            'value_model': value_model,
            'replay_buffer': buffer,
            'noise': noise,
            'curiosity': icm,
            'state_normalizer': state_normalizer,
            'goal_normalizer': goal_normalizer,
            'callbacks': [RLWandbCallback(project_name=config.get('wandb_project', 'PhoenX_RL'))],
            'save_dir': config.get('save_dir', 'Tune_Results'),
            'device': config.get('device', 'cuda'),
            # Hyperparameters (only pass if present)
            'discount': config.get('discount'),
            'policy_trace_decay': config.get('policy_trace_decay'),
            'value_trace_decay': config.get('value_trace_decay'),
            'tau': config.get('tau'),
            'alpha': config.get('alpha'),
            'auto_entropy_tuning': config.get('auto_entropy_tuning'),
            'alpha_lr': config.get('alpha_lr'),
            'batch_size': config.get('batch_size'),
            'grad_clip': (None if config.get('grad_clip') in [None, 'inf'] else config.get('grad_clip')),
            'warmup': config.get('warmup'),
            'N': config.get('N'),
            'action_epsilon': config.get('action_epsilon'),
            'noise_schedule': config.get('noise_schedule'),
            'target_noise': config.get('target_noise'),
            'target_noise_schedule': config.get('target_noise_schedule'),
            'target_noise_clip': config.get('target_noise_clip'),
            'actor_update_delay': config.get('actor_update_delay'),
            'gae_coefficient': config.get('gae_coefficient'),
            'policy_clip': config.get('policy_clip'),
            'value_clip': config.get('value_clip'),
            'value_loss_coefficient': config.get('value_loss_coefficient'),
            'entropy_coefficient': config.get('entropy_coefficient'),
            'entropy_schedule': config.get('entropy_schedule'),
            'kl_coefficient': config.get('kl_coefficient'),
            'kl_adapter': config.get('kl_adapter'),
            'normalize_advantages': config.get('normalize_advantages'),
            'reward_clip': config.get('reward_clip'),
            'log_level': config.get('log_level'),
        }
        # Remove None values
        common_kwargs = {k: v for k, v in common_kwargs.items() if v is not None}
        # Filter by agent signature
        sig = inspect.signature(agent_class.__init__)
        allowed = set(sig.parameters.keys()) - {'self'}
        agent_kwargs = {k: v for k, v in common_kwargs.items() if k in allowed}
        agent = agent_class(**agent_kwargs)

        for iteration in range(config['max_iterations']):
            # Prefer calling agent.train for one episode to ensure env interaction and metrics
            if hasattr(agent, 'train'):
                try:
                    train_sig = inspect.signature(agent.train)
                    train_kwargs = {}
                    if 'num_episodes' in train_sig.parameters:
                        train_kwargs['num_episodes'] = 1
                    if 'num_envs' in train_sig.parameters:
                        train_kwargs['num_envs'] = config.get('num_envs', 1)
                    if 'render_freq' in train_sig.parameters:
                        train_kwargs['render_freq'] = 0
                    #DEBUG
                    print('train_step called')
                    print(f'train_sig: {train_sig}')
                    print(f'train_kwargs: {train_kwargs}')
                    agent.train(**train_kwargs)
                except Exception:
                    # Fallback to a single learn step if train signature mismatches
                    try:
                        #DEBUG
                        print('learn called')
                        agent.learn()
                    except TypeError:
                        #DEBUG
                        print('learn called with goal_normalizer')
                        agent.learn(goal_normalizer=goal_normalizer)
            else:
                # Fallback to learn loop
                try:
                    #DEBUG
                    print('else learn called')
                    agent.learn()
                except TypeError:
                    #DEBUG
                    print('else learn called with goal_normalizer')
                    agent.learn(goal_normalizer=goal_normalizer)

            metrics = getattr(agent, '_train_step_config', None) or getattr(agent, '_train_episode_config', None) or {}
            if 'episode_reward' not in metrics:
                # Ensure the Tune metric exists even if no episode completed yet
                metrics['episode_reward'] = metrics.get('avg_reward', 0.0)
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
            
            def conditional_sampler(cfg_like, _depends_on=depends_on, _conditions=conditions):
                cfg = cfg_like.config if hasattr(cfg_like, 'config') else cfg_like
                parent_value = cfg.get(_depends_on)
                sub_params = _conditions.get(parent_value, {})  # Get matching sub-dict or empty
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

                    def layer_sampler(config, _value=value, _depends_on=depends_on):
                        num_layers = config.get(_depends_on, 0)
                        result = {}
                        for layer_num_str, layer_def in _value.items():
                            # Skip non-numeric keys (e.g., 'default')
                            try:
                                layer_num = int(layer_num_str)
                            except (TypeError, ValueError):
                                continue
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
                                        cond_depends_on = cond_value["depends_on"]
                                        dependent_key = cond_depends_on.split("_")[-1]
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
            metric='episode_reward',
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
            callbacks=[WandbLoggerCallback(project=user_config['wandb_project'], api_key_file='/workspaces/PhoenX_RL/src/app/wandb_api_key', log_config=True, upload_checkpoints=True)]
        )
    )

    results = tuner.fit()
    best_config = results.get_best_result().config
    return best_config, results