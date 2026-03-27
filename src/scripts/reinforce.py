from app.rl_agents import Reinforce
from app.env_wrapper import EnvWrapper
from app.models import StochasticDiscretePolicy, ValueModel
from app.normalizer import create_normalizer
from scripts.agent import infer_dim
from app.schedulers import ScheduleWrapper

def build(config: dict, env: EnvWrapper):
    # build policy
    policy_config = config['models']['policy']
    policy_config['env'] = env
    policy = StochasticDiscretePolicy(**policy_config)

    # # build value model if not None
    if config.get('models', {}).get('value', None):
        value_config = config['models']['value']
        value_config['env'] = env
        value = ValueModel(**value_config)
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

    # Create ActorCritic Agent
    return Reinforce(
                    policy=policy,
                    value=value,
                    discount=config['agent']['discount'],
                    state_normalizer=state_normalizer,
                    advantage_normalizer=advantage_normalizer,
                    entropy_coefficient=config['agent']['entropy_coefficient'],
                    entropy_schedule=ScheduleWrapper(type=config['entropy_schedule']['type'], **config['entropy_schedule']['config']) if config.get('entropy_schedule', None) else None,
                    save_dir=config['save_dir'],
                    device=config['device'],
    )
