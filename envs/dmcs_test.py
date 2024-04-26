import dm_control
from dm_control import suite
import gym
import numpy as np


# env = suite.load(domain_name="fish", task_name="swim", task_kwargs={'random': 0})
env = suite.load(domain_name="ball_in_cup", task_name="catch", task_kwargs={'random': 0})

print('action space', env.action_spec())
print('observation space', env.observation_spec())
time_step = env.reset()
print('time_step', time_step)
print('positions : ', env.physics.position())