from phoenx.builder import apply_model_config, create_intrinsic_motivation, create_normalizer
from phoenx.rl_agents import DDPG
from phoenx.env_wrapper import EnvWrapper
from phoenx.schedulers import ScheduleWrapper
from phoenx.noise import Noise


def build(config: dict, env: EnvWrapper) -> DDPG:
    """Build a DDPG agent from a config.

    Args:
        config (dict): The config to build the DDPG agent from.
        env (EnvWrapper): The environment to build the DDPG agent for.

    Returns:
        DDPG: The built DDPG agent.
    """
    # Canonical 'model:' schema (roots/trunk/branches)
    apply_model_config(config['agent']['config'], env, "DDPG")

    # create noise object if present in config
    if config['agent']['config'].get('noise', None):
        config['agent']['config']['noise'] = Noise.create_instance(config['agent']['config']['noise']['type'], **config['agent']['config']['noise']['config']) if config['agent']['config'].get('noise', None) else None
    else:
        config['agent']['config']['noise'] = None

    # create noise scheduler object if present in config
    if config['agent']['config'].get('noise_schedule', None):
        config['agent']['config']['noise_schedule'] = ScheduleWrapper(**config['agent']['config']['noise_schedule']) if config['agent']['config'].get('noise_schedule', None) else None
    else:
        config['agent']['config']['noise_schedule'] = None

    # create state normalizer object if present in config
    if config['agent']['config'].get('state_normalizer', None):
        config['agent']['config']['state_normalizer'] = create_normalizer(config['agent']['config']['state_normalizer'], env, config['env']['config']['obs_key'])
    else:
        config['agent']['config']['state_normalizer'] = None

    # create goal normalizer object if present in config
    if config['agent']['config'].get('goal_normalizer', None):
        config['agent']['config']['goal_normalizer'] = create_normalizer(config['agent']['config']['goal_normalizer'], env, config['env']['config']['goal_key'])
    else:
        config['agent']['config']['goal_normalizer'] = None

    # create reward normalizer object if present in config
    if config['agent']['config'].get('reward_normalizer', None):
        config['agent']['config']['reward_normalizer'] = create_normalizer(config['agent']['config']['reward_normalizer'], env)
    else:
        config['agent']['config']['reward_normalizer'] = None

    # create intrinsic motivation object if present in config
    if config['agent']['config'].get('intrinsic_motivation', None):
        config['agent']['config']['intrinsic_motivation'] = create_intrinsic_motivation(config['agent']['config']['intrinsic_motivation'], env, config['env']['config']['obs_key'])
    else:
        config['agent']['config']['intrinsic_motivation'] = None

    return DDPG(**config['agent']['config'])