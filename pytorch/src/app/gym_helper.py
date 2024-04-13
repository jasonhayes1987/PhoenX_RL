import gymnasium as gym
import numpy as np

## HER FUNCTIONALITY ##

def get_her_goal_functions(env:gym.Env):
    """Returns a list of goal functions for the HER algorithm."""

    spec_id = env.spec.id
    funcs = {
        'CarRacing-v2':{
            'desired_goal': car_racing_desired_goal,
            'achieved_goal': car_racing_achieved_goal,
            'reward': car_racing_reward
        }
    }

    return funcs[spec_id].values()

def car_racing_desired_goal(env):
    """Returns the desired goal for the CarRacing environment."""
    return np.array([len(env.get_wrapper_attr('track'))])

def car_racing_achieved_goal(env):
    """Returns the achieved goal for the CarRacing environment."""
    return np.array([env.get_wrapper_attr('tile_visited_count')])

def car_racing_reward(state_achieved_goal, next_state_achieved_goal, desired_goal):
    """Returns the reward for the CarRacing environment."""

    # if the state achieved goal is equivalent to the next states achieved goal,
    # no new track tiles were visited and therefore the reward is -0.1
    if state_achieved_goal == next_state_achieved_goal:
        return -0.1
    # else return reward based on desired goal
    return (1000/desired_goal) * (next_state_achieved_goal - state_achieved_goal)