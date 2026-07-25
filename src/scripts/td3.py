from scripts.agent import apply_model_config, create_intrinsic_motivation, create_normalizer, create_actor, create_critic
from app.rl_agents import TD3
from app.env_wrapper import EnvWrapper
from app.schedulers import ScheduleWrapper
from app.noise import Noise


def build(config: dict, env: EnvWrapper):
    # Canonical 'model:' schema (roots/trunk/branches) or legacy per-model keys
    if not apply_model_config(config['agent']['config'], env):
        # build policy
        config['agent']['config']['policy'] = create_actor(config['agent']['config']['policy'], env)

        # build critic model
        config['agent']['config']['critic'] = create_critic(config['agent']['config']['critic'], env)

        # build critic_b model if present in config
        config['agent']['config']['critic_b'] = create_critic(config['agent']['config']['critic_b'], env) if config['agent']['config'].get('critic_b', None) else None

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

    # create target noise object if present in config
    if config['agent']['config'].get('target_noise', None):
        config['agent']['config']['target_noise'] = Noise.create_instance(config['agent']['config']['target_noise']['type'], **config['agent']['config']['target_noise']['config']) if config['agent']['config'].get('target_noise', None) else None
    else:
        config['agent']['config']['target_noise'] = None

    # create target noise scheduler object if present in config
    if config['agent']['config'].get('target_noise_schedule', None):
        config['agent']['config']['target_noise_schedule'] = ScheduleWrapper(**config['agent']['config']['target_noise_schedule']) if config['agent']['config'].get('target_noise_schedule', None) else None
    else:
        config['agent']['config']['target_noise_schedule'] = None

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

    return TD3(**config['agent']['config'])