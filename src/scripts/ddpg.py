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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create DDPG Agent from config file')
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
actor_config = config['models']['actor']
actor_config['env'] = env
actor = ActorModel(**actor_config)

# # build critic
critic_config = config['models']['critic']
critic_config['env'] = env
critic = CriticModel(**critic_config)

# build replay buffer
config['replay_buffer']['config']['env'] = env
if config['replay_buffer']['class_name'] == 'ReplayBuffer':
    replay_buffer = ReplayBuffer(**config['replay_buffer']['config'])
elif config['replay_buffer']['class_name'] == 'PrioritizedReplayBuffer':
    replay_buffer = PrioritizedReplayBuffer(**config['replay_buffer']['config'])
else:
    raise ValueError(f"Invalid replay buffer class name: {config['replay_buffer']['class_name']}")

# create noise object if present in config
noise = Noise.create_instance(config['noise']['type'], **config['noise']['params']) if config.get('noise') else None

# create noise scheduler object if present in config
noise_schedule = ScheduleWrapper(**config["noise_schedule"]) if config.get("noise_schedule") else None

# create curiosity object if present in config
curiosity = ICM.create_instance(**config['curiosity']) if config.get('curiosity') else None

# create state normalizer object if present in config
if config['normalizers'].get('state', None):
    size = infer_dim(env, config['obs_key'])
    state_normalizer = Normalizer(size=size, **config['normalizers']['state'])
else:
    state_normalizer = None

# create goal normalizer object if present in config
if config['normalizers'].get('goal', None):
    size = infer_dim(env, config['goal_key'])
    goal_normalizer = Normalizer(size=size, **config['normalizers']['goal'])
else:
    goal_normalizer = None

# create callbacks object if present in config
callbacks = [callback_load(callback) for callback in config['callbacks']] if config.get('callbacks') else None

# Create DDPG Agent
agent_class = get_agent_class_from_type('DDPG')
ddpg = agent_class(
                env=env,
                actor_model=actor,
                critic_model=critic,
                replay_buffer=replay_buffer,
                discount=config['agent']['discount'],
                tau=config['agent']['tau'],
                action_epsilon=config['agent']['action_epsilon'],
                batch_size=config['agent']['batch_size'],
                noise=noise,
                noise_schedule=noise_schedule,
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
ddpg.save()

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