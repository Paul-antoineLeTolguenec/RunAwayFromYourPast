from config_env import config
import numpy as np
from wenv import Wenv
import envs

for name_id, conf in config.items():
    print('name_id : ', name_id)
    env = Wenv(env_id=name_id, **conf)
    print('coverage_idx', conf['coverage_idx'])
    print('matrix_coverage shape', env.matrix_coverage.shape)
        # print('observation space', env.observation_space)
        # print('action space', env.action_space)
        # # reset
        # obs, i = env.reset()
        # print('obs.shape', obs.shape)
        # print('i', i)
        # # step 
        # obs, reward, done, trunc, i = env.step(env.action_space.sample())
        # print('obs.shape', obs.shape)
        # print('i', i)