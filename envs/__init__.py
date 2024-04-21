from gym.envs.registration import register

# Maze
register(
    id='Maze-Easy',
    entry_point='envs.continuous_maze:Maze', 
    kwargs={'name': 'Easy', 'max_episode_steps': 200},
    max_episode_steps = 200
)
register(
    id='Maze-Ur',
    entry_point='envs.continuous_maze:Maze', 
    kwargs={'name': 'Ur', 'max_episode_steps': 200},
    max_episode_steps = 200
)
register(
    id='Maze-Hard',
    entry_point='envs.continuous_maze:Maze', 
    kwargs={'name': 'Hard', 'max_episode_steps': 200},
    max_episode_steps = 200
)
# DMCS
register(
    id='DMCS-Acrobot',
    entry_point='envs.dmcsw:DMCSWrapper', 
    kwargs={'domain_name': 'acrobot', 'task_name': 'swingup', 'max_episode_steps': 200, 'random': 0, 'render_mode': 'rgb_array'},
    max_episode_steps = 200
)
register(
    id='DMCS-Ball-in-cup',
    entry_point='envs.dmcsw:DMCSWrapper',
    kwargs={'domain_name': 'ball_in_cup', 'task_name': 'catch', 'max_episode_steps': 200, 'random': 0, 'render_mode': 'rgb_array'},
    max_episode_steps = 200
)
register(
    id='DMCS-Cart-k-Pole',
    entry_point='envs.dmcsw:DMCSWrapper',
    kwargs={'domain_name': 'cartpole', 'task_name': 'three_poles', 'max_episode_steps': 200, 'random': 0, 'render_mode': 'rgb_array'},
    max_episode_steps = 200
)
register(
    id='DMCS-Finger',
    entry_point='envs.dmcsw:DMCSWrapper',
    kwargs={'domain_name': 'finger', 'task_name': 'spin', 'max_episode_steps': 200, 'random': 0, 'render_mode': 'rgb_array'},
    max_episode_steps = 200
)
register(
    id='DMCS-Fish',
    entry_point='envs.dmcsw:DMCSWrapper',
    kwargs={'domain_name': 'fish', 'task_name': 'upright', 'max_episode_steps': 200, 'random': 0, 'render_mode': 'rgb_array'},
    max_episode_steps = 200
)


