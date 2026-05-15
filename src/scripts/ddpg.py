from app.rl_agents import DDPG
from app.env_wrapper import EnvWrapper
import gymnasium as gym
from app.models import DiscreteCritic, ContinuousCritic, ActorModel
from app.normalizer import create_normalizer
from scripts.agent import infer_dim, create_intrinsic_motivation
from app.schedulers import ScheduleWrapper
# from app.icm import ICM
from app.noise import Noise


def build(config: dict, env: EnvWrapper):
    # build policy
    policy_config = config['models']['policy']
    policy_config['env'] = env
    policy_config['lr_scheduler'] = ScheduleWrapper(**config['policy_lr_schedule']) if config.get('policy_lr_schedule', None) else None
    policy = ActorModel(**policy_config)

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

    # create noise object if present in config
    if config.get('noise', None):
        noise = Noise.create_instance(config['noise']['type'], **config['noise']['config']) if config.get('noise', None) else None
    else:
        noise = None

    # create noise scheduler object if present in config
    if config.get('noise_schedule', None):
        noise_schedule = ScheduleWrapper(**config['noise_schedule']) if config.get('noise_schedule', None) else None
    else:
        noise_schedule = None

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

    # create intrinsic motivation object if present in config
    # if config.get('intrinsic_motivation', None):
    #     im_type = config['intrinsic_motivation']['type']
    #     im_config = config['intrinsic_motivation']['config']
    #     if im_config.get('obs_normalizer', None):
    #         num_features = infer_dim(env, config['env']['config']['obs_key'])
    #         im_config['obs_normalizer']['config'].update({'num_features': num_features})
    #     im_config.update({
    #         'env': env,
    #         'reward_scheduler': ScheduleWrapper(**im_config['reward_scheduler']) if im_config.get('reward_scheduler', None) else None,
    #         'obs_normalizer': create_normalizer(im_config['obs_normalizer'] if im_config.get('obs_normalizer', None) else None),
    #         'intrinsic_reward_normalizer': create_normalizer(im_config['intrinsic_reward_normalizer'] if im_config.get('intrinsic_reward_normalizer', None) else None),
    #     })
    #     intrinsic_motivation = IntrinsicMotivation.create_instance(im_type, **im_config)
    # else:
    #     intrinsic_motivation = None
    intrinsic_motivation = create_intrinsic_motivation(config, env)

    ddpg_config = config['agent']['config']
    ddpg_config.update({
        'policy': policy,
        'critic': critic,
        'state_normalizer': state_normalizer,
        'goal_normalizer': goal_normalizer,
        'reward_normalizer': reward_normalizer,
        'noise': noise,
        'noise_schedule': noise_schedule,
        'intrinsic_motivation': intrinsic_motivation
    })
    return DDPG(**ddpg_config)