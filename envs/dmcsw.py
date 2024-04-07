from dm_control import suite
from gym import spaces
import gym 
import numpy as np


class DMCSWrapper(gym.Env):
    def __init__(self, domain_name, task_name, task_kwargs):
        self.env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs=task_kwargs)
        self.action_space = self.env.action_spec().shape[0]
        total_shape = sum(spec.shape[0] for spec in self.env.observation_spec().values())
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_shape,))
        self.reward = self.env.reward_spec()
        self.discount = self.env.discount_spec()
    
    def reset(self):
        time_step = self.env.reset()
        return np.concatenate(list(time_step.observation.values())), {}
    
    def step(self, action):
        time_step = self.env.step(action)
        obs = np.concatenate(list(time_step.observation.values()))
        reward = time_step.reward
        done = time_step.last()
        info = {}
        trunc = False
        return obs, reward, done, trunc, info

    def render(self):
        pass

    def close(self):
        pass



if __name__ == '__main__':
    env = DMCSWrapper(domain_name="acrobot", task_name="swingup", task_kwargs={'time_limit': 500, 'random': 0})
    print('action space', env.action_space)
    print('observation space', env.observation_space)
    print('reward', env.reward)
    print('discount', env.discount)
    obs, info = env.reset()
    print('obs', obs)
    for i in range(1):
        action = np.ones(env.action_space)
        obs, reward, done, trunc, info = env.step(action)
        print('obs', obs)
        print('reward', reward)
        if done:
            break
    env.close()

# env = suite.load(domain_name="acrobot", task_name="swingup", task_kwargs={'time_limit': 500, 'random': 0})
# print('action space', env.action_spec().shape)
# print('observation space', env.observation_spec())
# print('reward', env.reward_spec())
# print('discount', env.discount_spec())
# time_step = env.reset()
# # print('time_step', time_step)
# # print('observation', time_step.observation)
# # print('reward', time_step.reward)
# for i in range(1):
#     action = np.ones(env.action_spec().shape[0])
#     time_step = env.step(action)
#     print('time_step', time_step)
# #     print('time_step', time_step)
# #     print('observation', time_step.observation)
# #     print('reward', time_step.reward)
# #     if time_step.last():
# #         break
