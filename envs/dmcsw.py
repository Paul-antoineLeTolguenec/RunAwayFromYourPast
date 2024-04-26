from dm_control import suite
from gym import spaces
import gym 
import numpy as np


class DMCSWrapper(gym.Env):
    def __init__(self, domain_name, task_name, max_episode_steps, random, render_mode):
        self.env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs={'time_limit': max_episode_steps, 'random': random})
        self.render_mode = render_mode
        self.task_name = task_name
        self.domain_name = domain_name
        self.max_episode_steps = max_episode_steps
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.env.action_spec().shape[0],))
        total_shape = sum( self.try_spec(spec) for spec in self.env.observation_spec().values())
        if domain_name == 'finger':
            total_shape += self.env.physics.tip_position().shape[0]
        if domain_name == 'fish':
            total_shape += self.env.physics.position().shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_shape,))
        self.reward = self.env.reward_spec()
        self.discount = self.env.discount_spec()
        self.num_steps = 0
    def try_spec(self, spec):
        try:
            return spec.shape[0]
        except:
            return 1
    def reset(self):
        time_step = self.env.reset()
        self.num_steps = 0
        if 'finger' in self.domain_name:
            return np.concatenate([self.env.physics.tip_position().tolist() + np.concatenate(list(time_step.observation.values())).tolist()]), {}
        elif 'fish' in self.domain_name:
            return np.concatenate([self.env.physics.position().tolist() +np.concatenate([v if not 'upright' in k else np.array([v]) for k,v in time_step.observation.items()]).tolist()]), {}
        else :
            return np.concatenate(list(time_step.observation.values())), {}
    
    def step(self, action):
        self.num_steps += 1
        time_step = self.env.step(action)
        if 'finger' in self.domain_name:
            obs = np.concatenate([self.env.physics.tip_position().tolist() + np.concatenate(list(time_step.observation.values())).tolist()])
        elif 'fish' in self.domain_name:
            obs = np.concatenate([self.env.physics.position().tolist() +np.concatenate([v if not 'upright' in k else np.array([v]) for k,v in time_step.observation.items()]).tolist()])
        else:
            obs = np.concatenate(list(time_step.observation.values()))
        reward = time_step.reward
        done = (self.num_steps >= self.max_episode_steps)
        info = {}
        trunc = False
        return obs, reward, done, trunc, info

    def render(self):
        pass

    def close(self):
        pass



if __name__ == '__main__':
    import envs
    from config_env import config
    from wenv import Wenv
    for name_id, conf in config.items():
        if conf['type_id'] == 'dmcs': 
            print('name_id : ', name_id)
            env = Wenv(env_id=name_id)
    # print('action space', env.action_space)
    # print('observation space', env.observation_space)
    # obs, info = env.reset()
    # print('obs shape', obs.shape)
    # for i in range(10):
    #     action = np.ones(env.action_space.shape[0])
    #     obs, reward, done, trunc, info = env.step(action)
    #     print('obs', obs)
    #     print('reward', reward)
    #     print('done', done)
    #     if done:
    #         break
    # env.close()

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
