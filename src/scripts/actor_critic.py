import sys
import os
import argparse
import yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import gymnasium as gym
from app.agent_utils import *


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
    parser = argparse.ArgumentParser(description='Create ActorCritic Agent from config file')
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
policy = StochasticDiscretePolicy(**policy_config)

# # build critic
value_config = config['models']['value']
value_config['env'] = env
value = ValueModel(**value_config)

# create advantage normalizer object if present in config
if config.get('normalizers', {}).get('advantage', None):
    advantage_normalizer = Normalizer(size=1, **config['normalizers']['advantage'])
else:
    advantage_normalizer = None

# create state normalizer object if present in config
if config.get('normalizers', {}).get('state', None):
    size = infer_dim(env, config['obs_key'])
    state_normalizer = Normalizer(size=size, **config['normalizers']['state'])
else:
    state_normalizer = None

# create callbacks object if present in config
callbacks = [callback_load(callback) for callback in config['callbacks']] if config.get('callbacks') else None

# Create DDPG Agent
agent_class = get_agent_class_from_type('ActorCritic')
actor_critic = agent_class(
                env=env,
                policy=policy,
                value=value,
                discount=config['agent']['discount'],
                policy_trace_decay=config['agent']['policy_trace_decay'],
                value_trace_decay=config['agent']['value_trace_decay'],
                entropy_coefficient=config['agent']['entropy_coefficient'],
                entropy_schedule=ScheduleWrapper(config['entropy_schedule']) if config.get('entropy_schedule', None) else None,
                gae_coefficient=config['agent']['gae_coefficient'],
                trajectory_length=config['agent']['trajectory_length'],
                state_normalizer=state_normalizer,
                advantage_normalizer=advantage_normalizer,
                callbacks=callbacks,
                save_dir=config['save_dir'],
                device=config['device'],
                log_level=config['log_level'])

# # Save Agent
actor_critic.save()

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