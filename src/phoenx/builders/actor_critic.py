from phoenx.rl_agents import ActorCritic
from phoenx.env_wrapper import EnvWrapper
from phoenx.builder import apply_model_config, create_normalizer
from phoenx.normalizer import create_normalizer as normalizer_factory
from phoenx.schedulers import ScheduleWrapper


def build(config: dict, env: EnvWrapper) -> ActorCritic:
    """Build an ActorCritic agent from a config.

    Args:
        config (dict): The config to build the ActorCritic agent from.
        env (EnvWrapper): The environment to build the ActorCritic agent for.

    Returns:
        ActorCritic: The built ActorCritic agent.
    """
    agent_cfg = config['agent']['config']

    # Canonical 'model:' schema (roots/trunk/branches)
    apply_model_config(agent_cfg, env, "ActorCritic")

    # create state normalizer object if present in config
    if agent_cfg.get('state_normalizer', None):
        agent_cfg['state_normalizer'] = create_normalizer(
            agent_cfg['state_normalizer'], env, config['env']['config']['obs_key']
        )
    else:
        agent_cfg['state_normalizer'] = None

    # create goal normalizer object if present in config
    if agent_cfg.get('goal_normalizer', None):
        agent_cfg['goal_normalizer'] = create_normalizer(
            agent_cfg['goal_normalizer'], env, config['env']['config']['goal_key']
        )
    else:
        agent_cfg['goal_normalizer'] = None

    # create advantage normalizer object if present in config
    if agent_cfg.get('advantage_normalizer', None):
        # Advantage is scalar; do not infer dims from env obs (Isaac Dict spaces).
        adv_cfg = dict(agent_cfg['advantage_normalizer'])
        adv_cfg['config'] = dict(adv_cfg.get('config', {}))
        adv_cfg['config']['num_features'] = 1
        agent_cfg['advantage_normalizer'] = normalizer_factory(adv_cfg)
    else:
        agent_cfg['advantage_normalizer'] = None

    # create reward normalizer object if present in config
    if agent_cfg.get('reward_normalizer', None):
        agent_cfg['reward_normalizer'] = create_normalizer(agent_cfg['reward_normalizer'], env)
    else:
        agent_cfg['reward_normalizer'] = None

    # create entropy schedule object if present in config
    if agent_cfg.get('entropy_schedule', None):
        agent_cfg['entropy_schedule'] = ScheduleWrapper(**agent_cfg['entropy_schedule'])
    else:
        agent_cfg['entropy_schedule'] = None

    return ActorCritic(**agent_cfg)
