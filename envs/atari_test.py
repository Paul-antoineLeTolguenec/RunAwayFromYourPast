from config_env import config
import numpy as np
from wenv import Wenv
import envs
import time

env = Wenv('MontezumaRevengeNoFrameskip-v4', **config['MontezumaRevengeNoFrameskip-v4'])
obs, i = env.reset()
print(obs.shape, i)
env.render()
for i in range(100):
    obs, reward, done, trunc, info = env.step(env.action_space.sample())
    print(obs.shape, reward, done, trunc, info)
    env.render()
    time.sleep(0.1)
env.close()
