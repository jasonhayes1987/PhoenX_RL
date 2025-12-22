import sys
import os

# Use environment variable for IsaacLab path, with fallback to relative path
ISAACLAB_PATH = os.environ.get('ISAACLAB_PATH', os.path.join(os.path.dirname(__file__), '..', '..', 'IsaacLab', 'source'))
ISAACLAB_TASKS_PATH = os.path.join(ISAACLAB_PATH, 'isaaclab_tasks')

sys.path.append(ISAACLAB_PATH)
sys.path.append(ISAACLAB_TASKS_PATH)

print(f"ISAACLAB_PATH: {ISAACLAB_PATH}")
print(f"ISAACLAB_TASKS_PATH: {ISAACLAB_TASKS_PATH}")
print(f"Python path includes Isaac Lab: {'isaaclab' in str(sys.path)}")

def test_isaac_wrapper():
    """Test the IsaacSimWrapper functionality."""

    try:
        # Import our wrapper
        from env_wrapper import IsaacSimWrapper
        print("SUCCESS: IsaacSimWrapper import successful")

        # Test 1: Create IsaacSimWrapper with Cartpole environment
        cfg_string = "isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg:CartpoleEnvCfg"
        print(f"\nTesting with cfg: {cfg_string}")

        print("Creating IsaacSimWrapper...")
        wrappers = [
            {
                "type": "VectorNStepReward",
                "params": {
                    "n": 3,
                    "obs_key": 'policy',
                    "goal_key": None,
                    "ach_goal_key": None
                }
            }
        ]
        env_wrapper = IsaacSimWrapper(
            cfg=cfg_string,
            num_envs=2,
            wrappers=wrappers,
            render_mode='headless',
            seed=42,
        )
        print("SUCCESS: IsaacSimWrapper created successfully!")

        # Test basic properties
        print(f"Observation space: {env_wrapper.observation_space}")
        print(f"Action space: {env_wrapper.action_space}")
        print(f"Single observation space: {env_wrapper.single_observation_space}")
        print(f"Single action space: {env_wrapper.single_action_space}")

        # Test 2: Test environment reset
        print("\nTesting environment reset...")
        state, info = env_wrapper.reset()
        print("SUCCESS: Reset successful!")
        print(f"State shape: {state.shape if hasattr(state, 'shape') else type(state)}")
        print(f"Info keys: {list(info.keys())}")

        if 'n-step trajectory' in info:
            traj = info['n-step trajectory']
            print(f"Trajectory keys: {list(traj.keys())}")

        # Test 3: Test environment stepping
        print("\nTesting environment stepping...")
        import numpy as np

        # Get action space info
        action_space = env_wrapper.action_space
        print(f"Action space: {action_space}")

        # Generate a random action
        if hasattr(action_space, 'sample'):
            action = action_space.sample()
            print(f"Sampled action: {action}")
        else:
            # Manual random action for continuous space
            action = np.random.uniform(-1, 1, action_space.shape)
            print(f"Random action: {action}")

        # Format action for Isaac Sim (convert to tensor)
        formatted_action = env_wrapper.format_actions(action)
        print(f"Formatted action: {formatted_action}")

        # Step the environment
        next_states, rewards, dones, info = env_wrapper.step(formatted_action)
        print("SUCCESS: Step successful!")
        print(f"Next states shape: {next_states.shape if hasattr(next_states, 'shape') else type(next_states)}")
        print(f'Next states: {next_states}')
        print(f'Processing next states...')
        next_states = next_states['policy']
        print(f'Processed next states: {next_states}')
        print(f"Rewards: {rewards}")
        print(f"Dones: {dones}")
        print(f"Info: {info}")

        if 'n-step trajectory' in info:
            traj = info['n-step trajectory']
            print(f"Updated trajectory states shape: {traj['states'].shape}")

        # Test 4: Test cleanup
        print("\nTesting environment cleanup...")
        env_wrapper.close()
        print("SUCCESS: Environment closed successfully!")

        print("\nSUCCESS: All tests passed!")

    except Exception as e:
        print(f"FAILED: Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_isaac_wrapper()
