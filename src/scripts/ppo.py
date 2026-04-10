from app.rl_agents import PPO
from app.env_wrapper import EnvWrapper
from app.models import StochasticDiscretePolicy, StochasticContinuousPolicy, ValueModel
from app.normalizer import create_normalizer
from scripts.agent import infer_dim
from app.schedulers import ScheduleWrapper
from app.adaptive_kl import AdaptiveKL
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
    
    # build value model
    value_config = config['models']['value']
    value_config['env'] = env
    value_config['lr_scheduler'] = ScheduleWrapper(**config['value_lr_schedule']) if config.get('value_lr_schedule', None) else None
    value = ValueModel(**value_config)

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

    # create advantage normalizer object if present in config
    if config.get('normalizers', {}).get('advantage', None):
        config['normalizers']['advantage']['config'].update({'num_features': 1})
        advantage_normalizer = create_normalizer(config['normalizers']['advantage'])
    else:
        advantage_normalizer = None

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

    ppo_config = config['agent']['config']
    ppo_config.update({
        'policy': policy,
        'value': value,
        'state_normalizer': state_normalizer,
        'goal_normalizer': goal_normalizer,
        'advantage_normalizer': advantage_normalizer,
        'reward_normalizer': reward_normalizer,
        'entropy_schedule': ScheduleWrapper(**config['entropy_schedule']) if config.get('entropy_schedule', None) else None,
        'kl_adapter': AdaptiveKL(**config['kl_adapter']) if config.get('kl_adapter', None) else None,
        'policy_clip_schedule': ScheduleWrapper(**config['policy_clip_schedule']) if config.get('policy_clip_schedule', None) else None,
        'value_clip_schedule': ScheduleWrapper(**config['value_clip_schedule']) if config.get('value_clip_schedule', None) else None,
        'curiosity': curiosity
    })

    # Create PPO Agent
    return PPO(**ppo_config)