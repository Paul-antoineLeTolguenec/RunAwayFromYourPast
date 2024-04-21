from config_env import config
import numpy as np
from wenv import Wenv
import envs
import time
# from gym.spaces import Box
from gymnasium.spaces import Box

env = Wenv('PitfallNoFrameskip-v4', **config['PitfallNoFrameskip-v4'])
# env = Wenv('MontezumaRevengeNoFrameskip-v4', **config['MontezumaRevengeNoFrameskip-v4'])
# env = Wenv('Hopper-v3', **config['Hopper-v3'])

obs, i = env.reset()
print(obs.shape, i)
# env.render()
t0 = time.time()
done = False
t=0
# while not done:
for i in range(1000):
    t+=1
    obs, reward, done, trunc, info = env.step(env.action_space.sample())
    print(obs.shape, reward, done, trunc, info)
    print('Time:', t)
    env.render()
    time.sleep(0.1)
print('Time:', time.time() - t0)    
env.close()
