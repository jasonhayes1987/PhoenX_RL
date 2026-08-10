from phoenx.builder import apply_model_config, create_intrinsic_motivation, create_normalizer
from phoenx.rl_agents import SAC
from phoenx.env_wrapper import EnvWrapper
from phoenx.schedulers import ScheduleWrapper


def build(config: dict, env: EnvWrapper):
    # Canonical 'model:' schema (roots/trunk/branches)
    apply_model_config(config['agent']['config'], env, "SAC")

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

    # create curiosity object if present in config
    if config['agent']['config'].get('intrinsic_motivation', None):
        config['agent']['config']['intrinsic_motivation'] = create_intrinsic_motivation(config['agent']['config']['intrinsic_motivation'], env, config['env']['config']['obs_key'])
    else:
        config['agent']['config']['intrinsic_motivation'] = None

    # create entropy schedule object if present in config
    if config['agent']['config'].get('entropy_schedule', None):
        config['agent']['config']['entropy_schedule'] = ScheduleWrapper(**config['agent']['config']['entropy_schedule'])
    else:
        config['agent']['config']['entropy_schedule'] = None

    return SAC(**config['agent']['config'])