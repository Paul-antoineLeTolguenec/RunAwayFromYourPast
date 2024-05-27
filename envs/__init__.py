from gymnasium.envs.registration import register

# Maze
register(
    id='Maze-Easy-v0',
    entry_point='envs.continuous_maze:Maze', 
    kwargs={'name': 'Easy', 'max_episode_steps': 200},
    max_episode_steps = 200
)
register(
    id='Maze-Ur-v0',
    entry_point='envs.continuous_maze:Maze', 
    kwargs={'name': 'Ur', 'max_episode_steps': 200},
    max_episode_steps = 200
)
register(
    id='Maze-Hard-v0',
    entry_point='envs.continuous_maze:Maze', 
    kwargs={'name': 'Hard', 'max_episode_steps': 200},
    max_episode_steps = 200
)
# DMCS
register(
    id='DMCS-Acrobot-v0',
    entry_point='envs.dmcsw:DMCSWrapper', 
    kwargs={'domain_name': 'acrobot', 'task_name': 'swingup', 'max_episode_steps': 200, 'random': 0, 'render_mode': 'rgb_array'},
    max_episode_steps = 200
)
register(
    id='DMCS-Ball-in-cup-v0',
    entry_point='envs.dmcsw:DMCSWrapper',
    kwargs={'domain_name': 'ball_in_cup', 'task_name': 'catch', 'max_episode_steps': 200, 'random': 0, 'render_mode': 'rgb_array'},
    max_episode_steps = 200
)
register(
    id='DMCS-Cart-k-Pole-v0',
    entry_point='envs.dmcsw:DMCSWrapper',
    kwargs={'domain_name': 'cartpole', 'task_name': 'three_poles', 'max_episode_steps': 200, 'random': 0, 'render_mode': 'rgb_array'},
    max_episode_steps = 200
)
register(
    id='DMCS-Finger-v0',
    entry_point='envs.dmcsw:DMCSWrapper',
    kwargs={'domain_name': 'finger', 'task_name': 'spin', 'max_episode_steps': 200, 'random': 0, 'render_mode': 'rgb_array'},
    max_episode_steps = 200
)
register(
    id='DMCS-Fish-v0',
    entry_point='envs.dmcsw:DMCSWrapper',
    kwargs={'domain_name': 'fish', 'task_name': 'swim', 'max_episode_steps': 200, 'random': 0, 'render_mode': 'rgb_array'},
    max_episode_steps = 200
)


