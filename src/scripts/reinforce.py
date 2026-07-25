from app.rl_agents import Reinforce
from app.env_wrapper import EnvWrapper
from app.models import StochasticDiscreteHead, ValueHead
from app.normalizer import create_normalizer
from scripts.agent import apply_model_config, infer_dim
from app.schedulers import ScheduleWrapper

def build(config: dict, env: EnvWrapper):
    # Canonical 'model:' schema (roots/trunk/branches) or legacy per-model keys
    if apply_model_config(config['agent']['config'], env):
        policy = config['agent']['config'].pop('policy', None)
        value = config['agent']['config'].pop('value', None)
    else:
        # build policy
        policy_config = config['models']['policy']
        policy_config['env'] = env
        policy_config['temperature_schedule'] = ScheduleWrapper(**config['temperature_schedule']) if config.get('temperature_schedule', None) else None
        policy = StochasticDiscreteHead(**policy_config)

        # # build value model if not None
        if config.get('models', {}).get('value', None):
            value_config = config['models']['value']
            value_config['env'] = env
            value = ValueHead(**value_config)
        else:
            value = None

    # create state normalizer object if present in config
    if config.get('normalizers', {}).get('state', None):
        num_features = infer_dim(env, config['env']['config']['obs_key'])
        config['normalizers']['state']['config'].update({'num_features': num_features})
        state_normalizer = create_normalizer(config['normalizers']['state'])
    else:
        state_normalizer = None

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

    reinforce_config = config['agent']['config']
    reinforce_config.update({
        'policy': policy,
        'value': value,
        'entropy_schedule': ScheduleWrapper(**config['entropy_schedule']) if config.get('entropy_schedule', None) else None,
        'state_normalizer': state_normalizer,
        'advantage_normalizer': advantage_normalizer,
        'reward_normalizer': reward_normalizer,
    })

    # Create Reinforce Agent
    return Reinforce(**reinforce_config)
    # return Reinforce(
    #                 policy=policy,
    #                 value=value,
    #                 discount=config['agent']['discount'],
    #                 state_normalizer=state_normalizer,
    #                 advantage_normalizer=advantage_normalizer,
    #                 entropy_coefficient=config['agent']['entropy_coefficient'],
    #                 entropy_schedule=ScheduleWrapper(type=config['entropy_schedule']['type'], **config['entropy_schedule']['config']) if config.get('entropy_schedule', None) else None,
    #                 save_dir=config['save_dir'],
    #                 device=config['device'],
    # )
