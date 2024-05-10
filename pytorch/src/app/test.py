import gymnasium as gym
import gymnasium_robotics as gym_robo
# import wandb

def main():
    # wandb.login(key='758ac5ba01e12a3df504d2db2fec8ba4f391f7e6')
    # gym_robo.register_robotics_envs()

    env = gym.make('FetchPush-v2', max_episode_steps=50, render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(env, 'test/', episode_trigger=lambda i: i%1==0)

    episodes = 10


    for episode in range(episodes):
        done = False
        obs, _ = env.reset()
        while not done:
            obs, r, term, trunc, dict = env.step(env.action_space.sample())
            if term or trunc:
                done = True
    env.close()

if __name__ == '__main__':
    main()