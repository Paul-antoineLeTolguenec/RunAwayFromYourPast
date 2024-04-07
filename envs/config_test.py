from config_env import config
import numpy as np
from wenv import Wenv

# env = Wenv(env_id = 'Hopper-v3', **config['Hopper-v3'])
# env = Wenv(env_id = 'FetchReach-v3', **config['FetchReach-v3'])
# env = Wenv(env_id = 'Swimmer-v3', **config['Swimmer-v3'])
env = Wenv(env_id = 'Reacher-v4', **config['Reacher-v4'])

obs, i = env.reset(seed=0)
print(obs)
print(i)
print('obs shape', obs.shape)
for i in range(5):
    action = np.ones(env.action_space.shape[0])
    obs, reward, done, trunc, info = env.step(action)
    print('obs ', obs)
   
    if done:
        break