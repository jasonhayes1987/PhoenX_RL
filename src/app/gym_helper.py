import gymnasium as gym
import numpy as np

## HER/PER FUNCTIONALITY ##

def get_goal_envs():
    """Returns a list of envs that use goal states"""

    envs = []

def get_her_goal_functions(env:gym.Env):
    """Returns a list of goal functions for the HER algorithm."""

    spec_id = env.spec.id
    funcs = {
        'CarRacing-v2':{
            'desired_goal': car_racing_desired_goal,
            'achieved_goal': car_racing_achieved_goal,
            'reward': car_racing_reward
        },

        'Reacher-v4':{
            'desired_goal': reacher_desired_goal,
            'achieved_goal': reacher_achieved_goal,
            'reward': reacher_reward
        },

        # 'Pusher-v5':{ 
        #     'desired_goal': pusher_desired_goal,
        #     'achieved_goal': pusher_achieved_goal,
        #     'reward': pusher_reward
        # }

        'FetchReach-v2':{
            'desired_goal': fetch_reach_desired_goal,
            'achieved_goal': fetch_reach_achieved_goal,
            'reward': fetch_reach_reward
        },

        'FetchPickAndPlace-v2':{
            'desired_goal': fetch_pick_place_desired_goal,
            'achieved_goal': fetch_pick_place_achieved_goal,
            'reward': fetch_pick_place_reward
        },

        'FetchPush-v2':{
            'desired_goal': fetch_push_desired_goal,
            'achieved_goal': fetch_push_achieved_goal,
            'reward': fetch_push_reward
        },

        'FetchSlide-v2':{
            'desired_goal': fetch_slide_desired_goal,
            'achieved_goal': fetch_slide_achieved_goal,
            'reward': fetch_slide_reward
        },
    }

    return funcs[spec_id].values()

 
def car_racing_desired_goal(env):
    """Returns the desired goal for the CarRacing environment."""
    return np.array([len(env.get_wrapper_attr('track'))])

def car_racing_achieved_goal(env):
    """Returns the achieved goal for the CarRacing environment."""
    return np.array([env.get_wrapper_attr('tile_visited_count')])

def car_racing_reward(env, action, state_achieved_goal, next_state_achieved_goal, desired_goal, tolerance):
    diff = desired_goal - next_state_achieved_goal
    # if diff <= tolerance:
        # print('within tolerance')
    if diff <= tolerance:
        return 0,1 
    else:
        return -1,0

def reacher_desired_goal(env):
    """Returns the desired goal for the Reacher Mujoco environment."""
    return np.array([0.0, 0.0, 0.0])

def reacher_achieved_goal(env):
    return env.get_wrapper_attr("_get_obs")()[8::]

def reacher_reward(env, action, state_achieved_goal, next_state_achieved_goal, desired_goal, tolerance):
    distance = np.linalg.norm(desired_goal - next_state_achieved_goal)
    if distance <= tolerance:
        return 0,1
    else:
        return -1,0

# def pusher_desired_goal(env):
#     return env.get_wrapper_attr("get_body_com")("goal")

# def pusher_achieved_goal(env):
#     return env.get_wrapper_attr("get_body_com")("object")

# def pusher_reward(env, action, state_achieved_goal, next_state_achieved_goal, desired_goal, tolerance):
#     goal_distance = np.linalg.norm(next_state_achieved_goal - desired_goal)
#     arm_distance = np.linalg.norm(next_state_achieved_goal - env.get_wrapper_attr("get_body_com")("tips_arm"))
#     # add distances
#     total_distance = goal_distance + arm_distance
#     # print(f'distance = {total_distance}')
#     if total_distance <= tolerance:
#         # print(f'distance within in tolerance')
#         return 0,1
#     else:
#         return -1,0

def fetch_reach_desired_goal(env):
    return env.get_wrapper_attr("_get_obs")()['desired_goal']

def fetch_reach_achieved_goal(env):
    return env.get_wrapper_attr("_get_obs")()['achieved_goal']

def fetch_reach_reward(env, action=None, state_achieved_goal=None, next_state_achieved_goal=None, desired_goal=None, tolerance=None):
    
    distance = np.linalg.norm(next_state_achieved_goal - desired_goal, axis=-1)
    reward = env.get_wrapper_attr("compute_reward")(next_state_achieved_goal, desired_goal, None)

    if distance <= tolerance:
        return reward, 1
    else:
        return reward, 0
    
def fetch_pick_place_desired_goal(env):
    return env.get_wrapper_attr("_get_obs")()['desired_goal']

def fetch_pick_place_achieved_goal(env):
    return env.get_wrapper_attr("_get_obs")()['achieved_goal']

def fetch_pick_place_reward(env, action=None, state_achieved_goal=None, next_state_achieved_goal=None, desired_goal=None, tolerance=None):
    
    distance = np.linalg.norm(next_state_achieved_goal - desired_goal, axis=-1)
    reward = env.get_wrapper_attr("compute_reward")(next_state_achieved_goal, desired_goal, None)

    if distance <= tolerance:
        return reward, 1
    else:
        return reward, 0

def fetch_push_desired_goal(env):
    return env.get_wrapper_attr("_get_obs")()['desired_goal']

def fetch_push_achieved_goal(env):
    return env.get_wrapper_attr("_get_obs")()['achieved_goal']

def fetch_push_reward(env, action=None, state_achieved_goal=None, next_state_achieved_goal=None, desired_goal=None, tolerance=None):
    
    distance = np.linalg.norm(next_state_achieved_goal - desired_goal, axis=-1)
    # print(f'distance: {distance}')
    reward = env.get_wrapper_attr("compute_reward")(next_state_achieved_goal, desired_goal, None)

    if distance <= tolerance:
        return reward, 1
    else:
        return reward, 0

def fetch_slide_desired_goal(env):
    return env.get_wrapper_attr("_get_obs")()['desired_goal']

def fetch_slide_achieved_goal(env):
    return env.get_wrapper_attr("_get_obs")()['achieved_goal']

def fetch_slide_reward(env, action=None, state_achieved_goal=None, next_state_achieved_goal=None, desired_goal=None, tolerance=None):
    
    #DEBUG
    # print(f'next state achieved goal: {next_state_achieved_goal}')
    # print(f'desired goal: {desired_goal}')
    distance = np.linalg.norm(next_state_achieved_goal - desired_goal, axis=-1)
    reward = env.get_wrapper_attr("compute_reward")(next_state_achieved_goal, desired_goal, None)
    #DEBUG
    # print(f'distance: {distance}')
    # print(f'reward: {reward}')

    if distance <= tolerance:
        return reward, 1
    else:
        return reward, 0