import numpy as np
config = {
##########################       MAZE       ############################################
"Maze-Easy": { 
    'type_id': 'maze',
    'kwargs': {},
    'coverage_idx': np.array([0,1]),
    'render_settings': {'x_lim': [-1, 1], 'y_lim': [-1, 1]}
},
"Maze-Ur": { 
    'type_id': 'maze',
    'kwargs': {},
    'coverage_idx': np.array([0,1]),
    'render_settings': {'x_lim': [-1, 1], 'y_lim': [-1, 1]}
},
"Maze-Hard": { 
    'type_id': 'maze',
    'kwargs': {},
    'coverage_idx': np.array([0,1]),
    'render_settings': {'x_lim': [-1, 1], 'y_lim': [-1, 1]}
},
##########################       ROBOTICS       ############################################
"FetchReach-v3": {
    'type_id': 'robotics',
    'kwargs': {'render_mode': 'rgb_array', 
               'max_episode_steps': 50},
    'coverage_idx': np.array([0,1,2]),
    'render_settings': {'x_lim': [-1, 1], 'y_lim': [-1, 1], 'z_lim': [0, 1]}
},
"FetchPush-v3": {
    'type_id': 'robotics',
    'kwargs': {'render_mode': 'rgb_array', 
               'max_episode_steps': 50},
    'coverage_idx': np.array([0,1,2,3,4]),
    'render_settings': {'x_lim': [-1, 1], 'y_lim': [-1, 1], 'z_lim': [0, 1]}
},
"FetchSlide-v3": {
    'type_id': 'robotics',
    'kwargs': {'render_mode': 'rgb_array', 
               'max_episode_steps': 50},
    'coverage_idx': np.array([0,1,2,3,4]),
    'render_settings': {'x_lim': [-1, 1], 'y_lim': [-1, 1], 'z_lim': [0, 1]}
},
##########################       DMCS       ############################################
#  Acrobot,
"acrobot": {
    'type_id': 'dmcs',
    'kwargs': {'render_mode': 'rgb_array', 
               'max_episode_steps': 500},
    'coverage_idx': np.array([0,1]),
    'render_settings': {'x_lim': [-1, 1], 'y_lim': [-1, 1]}
},
#  Ball-in-cup, 
#  Cart-pole, 
#  Finger, 
#  Fish

##########################       ATARI       ############################################
# MontezumaRevengeNoFrameskip-v4
# PitfallNoFrameskip-v4

##########################       MUJOCO       ############################################

"HalfCheetah-v3": {
    'type_id': 'mujoco',
    'kwargs': {'render_mode': 'rgb_array', 
               'reset_noise_scale': 0.0, 
               'exclude_current_positions_from_observation': False, 
               'max_episode_steps': 1000},
    'coverage_idx': np.array([0,1]),
    'render_settings': {'x_lim': [-30, 30], 'y_lim': [-1, 1]}
},
"Hopper-v3": {
    'type_id': 'mujoco',
    'kwargs': {'render_mode': 'rgb_array', 
               'reset_noise_scale': 0.0, 
               'exclude_current_positions_from_observation': False, 
               'max_episode_steps': 1000, 
               'terminate_when_unhealthy': True},
    'coverage_idx': np.array([0,1]),
    'render_settings': {'x_lim': [-20, 20], 'y_lim': [0, 2]}
},
"Ant-v3": {
    'type_id': 'mujoco',
    'kwargs': {'render_mode': 'rgb_array', 
               'reset_noise_scale': 0.0, 
               'exclude_current_positions_from_observation': False, 
               'max_episode_steps': 1000,
               'terminate_when_unhealthy': True},
    'coverage_idx': np.array([0,1,2]),
    'render_settings': {'x_lim': [-10, 10], 'y_lim': [-10, 10], 'z_lim': [0, 2]}
},
"Walker2d-v3": {
    'type_id': 'mujoco',
    'kwargs': {'render_mode': 'rgb_array', 
               'reset_noise_scale': 0.0, 
               'exclude_current_positions_from_observation': False, 
               'max_episode_steps': 1000,
               'terminate_when_unhealthy': True},
    'coverage_idx': np.array([0,1]),
    'render_settings': {'x_lim': [-20, 20], 'y_lim': [0, 2]}
},
"Humanoid-v3": {
    'type_id': 'mujoco',
    'kwargs': {'render_mode': 'rgb_array', 
               'reset_noise_scale': 0.0, 
               'exclude_current_positions_from_observation': False, 
               'max_episode_steps': 1000,
               'terminate_when_unhealthy': True},
    'coverage_idx': np.array([0,1,2]),
    'render_settings': {'x_lim': [-5, 5], 'y_lim': [-5, 5], 'z_lim': [0, 2]}
},
"HumanoidStandup-v4": {
    'type_id': 'mujoco',
    'kwargs': {'render_mode': 'rgb_array', 
               'max_episode_steps': 1000},
    'coverage_idx': np.array([0,1,2]),
    'render_settings': {'x_lim': [-10, 10], 'y_lim': [-10, 10], 'z_lim': [0, 2]}
},
"Reacher-v4": {
    'type_id': 'mujoco',
    'kwargs': {'render_mode': 'rgb_array', 
               'max_episode_steps': 50},
    'coverage_idx': np.array([8,9]),
    'render_settings': {'x_lim': [-0.5, 0.5], 'y_lim': [-0.5, 0.5]}
},
"Swimmer-v3": {
    'type_id': 'mujoco',
    'kwargs': {'render_mode': 'rgb_array', 
               'reset_noise_scale': 0.0, 
               'exclude_current_positions_from_observation': False, 
               'max_episode_steps': 500},
    'coverage_idx': np.array([0,1]),
    'render_settings': {'x_lim': [-5, 5], 'y_lim': [-5, 5]}
},

}