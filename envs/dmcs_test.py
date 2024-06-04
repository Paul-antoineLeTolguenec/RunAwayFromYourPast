import dm_control
from dm_control import suite
import gymnasium as gym
import numpy as np


env = suite.load(domain_name="fish", task_name="swim", task_kwargs={'random': 0})
# env = suite.load(domain_name="ball_in_cup", task_name="catch", task_kwargs={'random': 0})

time_step = env.reset()
done = False
t = 0
while not done:
    t += 1
    action = np.random.uniform(-1, 1, env.action_spec().shape)
    time_step = env.step(action)
    done = time_step.last()
    print(f"Step: {t}, Reward: {time_step.reward}, Done: {done}")