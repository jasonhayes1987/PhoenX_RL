import sys
import os
import argparse
import yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import gymnasium as gym
from app.agent_utils import *
from app.icm import ICM


def load_config(config_file: str) -> dict:
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def infer_dim(env, key=None):
    space = env.single_observation_space
    if isinstance(space, gym.spaces.Dict):
        if key is None:
            raise ValueError(
                f"Observation space is Dict, but no key provided. "
                f"Available keys: {list(space.spaces.keys())}"
            )
        if key not in space.spaces:
            raise KeyError(
                f"Key '{key}' not in observation space. "
                f"Available keys: {list(space.spaces.keys())}"
            )
        return int(np.prod(space.spaces[key].shape))
    return int(np.prod(space.shape))

def create_normalizer(config, key, env):
    normalizers_config = config.get('normalizers', {})
    if normalizers_config.get(key):
        if key == 'state':
            env_key = config['obs_key']
        elif key == 'goal':
            env_key = config['goal_key']
        else:
            raise ValueError(f"Invalid normalizer key: {key}")
        size = infer_dim(env, env_key)
        return Normalizer(size=size, **normalizers_config[key])
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create TD3 Agent from config file')
    parser.add_argument('--config_file', type=str, required=True, help='Path to the agent configuration file (.yml)')
    args = parser.parse_args()
    config = load_config(args.config_file)
    print(config)

    # Create env object using correct wrapper
    if config['env']['type'] == 'isaacsim':
        #DEBUG
        print(f"Creating IsaacSimWrapper with config: {config['env']['config']}")
        env = IsaacSimWrapper(**config['env']['config'])
        print(f"Created IsaacSimWrapper with config: {config['env']['config']}")
    elif config['env']['type'] == 'gymnasium':
        env = GymnasiumWrapper(**config['env']['config'])
        print(f"Created GymnasiumWrapper with config: {config['env']['config']}")
    else:
        raise ValueError(f"Invalid environment type: {config['env']['type']}")



# # build actor
policy_config = config['models']['policy']
policy_config['env'] = env
policy_config['lr_scheduler'] = ScheduleWrapper(config["schedulers"]["policy_lr_schedule"]['type'], config["schedulers"]["policy_lr_schedule"]['params']) if config.get("schedulers", {}).get("policy_lr_schedule") else None
policy_model_class = select_policy_model(env)
policy = policy_model_class(**policy_config)

# # build critics
critic_a_config = config['models']['critic_a']
critic_a_config['env'] = env
critic_a_config['lr_scheduler'] = ScheduleWrapper(config["schedulers"]["critic_a_lr_schedule"]['type'], config["schedulers"]["critic_a_lr_schedule"]['params']) if config.get("schedulers", {}).get("critic_a_lr_schedule") else None
# critic_a = ContinuousCritic(**critic_a_config)
if config.get('models').get('critic_b', None):
    critic_b_config = config['models']['critic_b']
    critic_b_config['env'] = env
    critic_b_config['lr_scheduler'] = ScheduleWrapper(config["schedulers"]["critic_b_lr_schedule"]['type'], config["schedulers"]["critic_b_lr_schedule"]['params']) if config.get("schedulers", {}).get("critic_b_lr_schedule") else None
    # critic_b = ContinuousCritic(**critic_b_config)
else: critic_b_config = None
critic_model_class = select_critic_model(env)
critic_a = critic_model_class(**critic_a_config)
if critic_b_config:
    critic_b = critic_model_class(**critic_b_config)
else:
    critic_b = critic_a.clone(copy_weights=False)

# build value model
# value_config = config['models']['value']
# value_config['env'] = env
# value_config['lr_scheduler'] = ScheduleWrapper(config["schedulers"]["value_lr_schedule"]['type'], config["schedulers"]["value_lr_schedule"]['params']) if config.get("schedulers", {}).get("value_lr_schedule") else None
# value = ValueModel(**value_config)

# build replay buffer
config['replay_buffer']['config']['env'] = env
if config['replay_buffer']['type'] == 'ReplayBuffer':
    replay_buffer = ReplayBuffer(**config['replay_buffer']['config'])
elif config['replay_buffer']['type'] == 'PrioritizedReplayBuffer':
    replay_buffer = PrioritizedReplayBuffer(**config['replay_buffer']['config'])
else:
    raise ValueError(f"Invalid replay buffer type: {config['replay_buffer']['type']}")

# create curiosity object if present in config
if config.get('curiosity'):
    config['curiosity']['env'] = env
    if config['curiosity'].get('reward_scheduler'):
        config['curiosity']['reward_scheduler'] = ScheduleWrapper(config['curiosity']['reward_scheduler']['type'], config['curiosity']['reward_scheduler']['params'])
    else:
        config['curiosity']['reward_scheduler'] = None
    curiosity = ICM.create_instance(**config['curiosity'])
else:
    curiosity = None

# create state normalizer object if present in config
state_normalizer = create_normalizer(config, 'state', env)
goal_normalizer = create_normalizer(config, 'goal', env)

# create callbacks object if present in config
callbacks = [callback_load(callback) for callback in config['callbacks']] if config.get('callbacks') else None

# Create DDPG Agent
agent_class = get_agent_class_from_type('SAC')
sac = agent_class(
                env=env,
                policy_model=policy,
                critic_model_a=critic_a,
                critic_model_b=critic_b,
                replay_buffer=replay_buffer,
                discount=config['agent']['discount'],
                tau=config['agent']['tau'],
                alpha=config['agent']['alpha'],
                auto_entropy_tuning=config['agent']['auto_entropy_tuning'],
                alpha_lr=config['agent']['alpha_lr'],
                batch_size=config['agent']['batch_size'],
                grad_clip=config['agent']['grad_clip'],
                warmup=config['agent']['warmup'],
                N=config['agent']['N'],
                curiosity=curiosity,
                state_normalizer=state_normalizer,
                goal_normalizer=goal_normalizer,
                obs_key=config['obs_key'],
                goal_key=config['goal_key'],
                achieved_goal_key=config['achieved_goal_key'],
                callbacks=callbacks,
                save_dir=config['save_dir'],
                device=config['device'],
                log_level=config['log_level'])

# # Save Agent
sac.save()

# Set train config
train_config = config['train_config']
train_config_path = config['save_dir'] + 'train_config.json'
with open(train_config_path, 'w') as f:
    json.dump(train_config, f)

# Set test config
test_config = config['test_config']
test_config_path = config['save_dir'] + 'test_config.json'
with open(test_config_path, 'w') as f:
    json.dump(test_config, f)