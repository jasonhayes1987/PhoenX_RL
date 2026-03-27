from app.rl_agents import ActorCritic
from app.env_wrapper import EnvWrapper
from app.models import StochasticDiscretePolicy, StochasticContinuousPolicy, ValueModel
from app.normalizer import create_normalizer
from scripts.agent import infer_dim
from app.schedulers import ScheduleWrapper

def build(config: dict, env: EnvWrapper):
    # build policy
    policy_config = config['models']['policy']
    policy_config['env'] = env
    if config['models']['policy']['distribution'] in ['categorical']:
        policy = StochasticDiscretePolicy(**policy_config)
    elif config['models']['policy']['distribution'] in ['beta', 'kumaraswamy', 'normal']:
        policy = StochasticContinuousPolicy(**policy_config)
    else:
        raise ValueError(f"Invalid distribution: {config['models']['policy']['distribution']}")

    # # build value model
    value_config = config['models']['value']
    value_config['env'] = env
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

    ac_config = config['agent']['config']
    ac_config.update({
        'policy': policy,
        'value': value,
        'entropy_schedule': ScheduleWrapper(**config['entropy_schedule']) if config.get('entropy_schedule', None) else None,
        'state_normalizer': state_normalizer,
        'goal_normalizer': goal_normalizer,
        'advantage_normalizer': advantage_normalizer,

    })

    # Create ActorCritic Agent
    return ActorCritic(**ac_config)
    # return ActorCritic(
    #                 policy=policy,
    #                 value=value,
    #                 discount=config['agent']['discount'],
    #                 entropy_coefficient=config['agent']['entropy_coefficient'],
    #                 entropy_schedule=ScheduleWrapper(**config['entropy_schedule']) if config.get('entropy_schedule', None) else None,
    #                 gae_coefficient=config['agent']['gae_coefficient'],
    #                 state_normalizer=state_normalizer,
    #                 goal_normalizer=goal_normalizer,
    #                 advantage_normalizer=advantage_normalizer,
    #                 policy_grad_clip=config['agent']['policy_grad_clip'],
    #                 value_grad_clip=config['agent']['value_grad_clip'],
    #                 value_coef=config['agent']['value_coef'],
    #                 save_dir=config['save_dir'],
    #                 device=config['device'],
    #                 log_level=config['agent'].get('log_level', 'INFO'),
    #                 **dict(config['agent'].get('kwargs', {}))
    # )
