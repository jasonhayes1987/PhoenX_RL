from app.rl_agents import SAC
from app.env_wrapper import EnvWrapper
import gymnasium as gym
from app.models import StochasticDiscretePolicy, StochasticContinuousPolicy, ContinuousCritic, DiscreteCritic
from app.normalizer import create_normalizer
from scripts.agent import infer_dim
from app.schedulers import ScheduleWrapper
from app.icm import ICM


def build(config: dict, env: EnvWrapper):
    # build policy
    policy_config = config['models']['policy']
    policy_config['env'] = env
    policy_config['lr_scheduler'] = ScheduleWrapper(**config['policy_lr_schedule']) if config.get('policy_lr_schedule', None) else None
    if config['models']['policy']['distribution'] in ['categorical']:
        policy_config['temperature_schedule'] = ScheduleWrapper(**config['temperature_schedule']) if config.get('temperature_schedule', None) else None
        policy = StochasticDiscretePolicy(**policy_config)
    elif config['models']['policy']['distribution'] in ['beta', 'kumaraswamy', 'normal']:
        policy = StochasticContinuousPolicy(**policy_config)
    else:
        raise ValueError(f"Invalid distribution: {config['models']['policy']['distribution']}")

    # build critic model
    critic_config = config['models']['critic']
    critic_config['env'] = env
    critic_config['lr_scheduler'] = ScheduleWrapper(**config['critic_lr_schedule']) if config.get('critic_lr_schedule', None) else None
    if isinstance(env.single_action_space, gym.spaces.Discrete):
        critic = DiscreteCritic(**critic_config)
    elif isinstance(env.single_action_space, gym.spaces.Box):
        critic = ContinuousCritic(**critic_config)
    else:
        raise ValueError(f"Invalid action space: {env.single_action_space}")

    # build critic_b model if present in config
    critic_b_config = config['models']['critic_b'] if config.get('models', {}).get('critic_b', None) else None
    if critic_b_config:
        critic_b_config['env'] = env
        critic_b_config['lr_scheduler'] = ScheduleWrapper(**config['critic_b_lr_schedule']) if config.get('critic_b_lr_schedule', None) else None
        if isinstance(env.single_action_space, gym.spaces.Discrete):
            critic_b = DiscreteCritic(**critic_b_config)
        elif isinstance(env.single_action_space, gym.spaces.Box):
            critic_b = ContinuousCritic(**critic_b_config)
        else:
            raise ValueError(f"Invalid action space: {env.single_action_space}")
    else:
        critic_b = None

    # create state normalizer object if present in config
    if config.get('normalizers', {}).get('state', None):
        num_features = infer_dim(env, config['env']['config']['obs_key'])
        config['normalizers']['state']['config'].update({'num_features': num_features})
        state_normalizer = create_normalizer(config['normalizers']['state'])
    else:
        state_normalizer = None

    # create goal normalizer object if present in config
    if config.get('normalizers', {}).get('goal', None):
        num_features = infer_dim(env, config['env']['config']['goal_key'])
        config['normalizers']['goal']['config'].update({'num_features': num_features})
        goal_normalizer = create_normalizer(config['normalizers']['goal'])
    else:
        goal_normalizer = None

    # create reward normalizer object if present in config
    if config.get('normalizers', {}).get('reward', None):
        reward_normalizer = create_normalizer(config['normalizers']['reward'])
    else:
        reward_normalizer = None

    # create curiosity object if present in config
    if config.get('curiosity', None):
        config['curiosity'].update({
            'env': env,
            'reward_scheduler': ScheduleWrapper(**config['reward_scheduler']) if config.get('reward_scheduler', None) else None
        })
        curiosity = ICM(**config['curiosity'])
    else:
        curiosity = None

    sac_config = config['agent']['config']
    sac_config.update({
        'policy': policy,
        'critic': critic,
        'critic_b': critic_b,
        'state_normalizer': state_normalizer,
        'goal_normalizer': goal_normalizer,
        'reward_normalizer': reward_normalizer,
        'curiosity': curiosity
    })
    return SAC(**sac_config)