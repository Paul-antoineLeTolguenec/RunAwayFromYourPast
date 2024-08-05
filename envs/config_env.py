import numpy as np
config = {
##########################       MAZE       ############################################
"Maze-Easy-v0": { 
    'type_id': 'maze',
    'kwargs': {'max_episode_steps': 200},
    'coverage_idx': np.array([0,1]),
    'render_settings': {'x_lim': [-1, 1], 'y_lim': [-1, 1]}
},
"Maze-Ur-v0": { 
    'type_id': 'maze',
    'kwargs': {'max_episode_steps': 200},
    'coverage_idx': np.array([0,1]),
    'render_settings': {'x_lim': [-1, 1], 'y_lim': [-1, 1]}
},
"Maze-Hard-v0": { 
    'type_id': 'maze',
    'kwargs': {'max_episode_steps': 200},
    'coverage_idx': np.array([0,1]),
    'render_settings': {'x_lim': [-1, 1], 'y_lim': [-1, 1]}
},
##########################       ROBOTICS       ############################################
"FetchReach-v1": {
    'type_id': 'robotics',
    'kwargs': {'render_mode': 'rgb_array', 
               'max_episode_steps': 50},
    'coverage_idx': np.array([0,1,2]),
    'render_settings': {'x_lim': [0, 2], 'y_lim': [-2, 2], 'z_lim': [0, 5]}
},
"FetchPush-v2": {
    'type_id': 'robotics',
    'kwargs': {'render_mode': 'rgb_array', 
               'max_episode_steps': 50},
    'coverage_idx': np.array([3,4]),
    'render_settings': {'x_lim': [0,2], 'y_lim': [0,2]}
},
"FetchSlide-v2": {
    'type_id': 'robotics',
    'kwargs': {'render_mode': 'rgb_array', 
               'max_episode_steps': 50},
    'coverage_idx': np.array([3,4]),
    'render_settings': {'x_lim': [0,2], 'y_lim': [0,2]}
},
##########################       DMCS       ############################################
#  Acrobot,
"DMCS-Acrobot-v0": {
    'type_id': 'dmcs',
    'kwargs': {
        'render_mode': 'rgb_array', 
        'max_episode_steps': 1000},
    'coverage_idx': np.array([0,1,2,3]),
    'render_settings': {'x_lim': [-1, 1], 'y_lim': [-1, 1], 'z_lim': [-1, 1], 'w_lim': [-1, 1]}
},
#  Ball-in-cup, 
"DMCS-Ball-in-cup-v0": {
    'type_id': 'dmcs',
    'kwargs': {'render_mode': 'rgb_array', 
               'max_episode_steps': 1000},
    'coverage_idx': np.array([0,1,2,3]),
    'render_settings': {'x_lim': [-1, 1], 'y_lim': [-1, 1], 'z_lim': [-1, 1], 'w_lim': [-1, 1]}
},
#  Cart-pole, 
"DMCS-Cart-k-Pole-v0": {
    'type_id': 'dmcs',
    'kwargs': {'render_mode': 'rgb_array', 
               'max_episode_steps': 1000},
    'coverage_idx': np.array([4,5,6,7]),
    'render_settings': {'x_lim': [-1, 1], 'y_lim': [-1, 1], 'z_lim': [-1, 1], 'w_lim': [-1, 1]}
},
#  Finger, 
"DMCS-Finger-v0": {
    'type_id': 'dmcs',
    'kwargs': {'render_mode': 'rgb_array', 
               'max_episode_steps': 1000},
    'coverage_idx': np.array([2,3,4,5]),
    'render_settings': {'x_lim': [-2.5, 2.5], 'y_lim': [-2.5, 2.5], 'z_lim': [-2.5, 2.5], 'w_lim': [-2.5, 2.5]}
},
#  Fish
"DMCS-Fish-v0": {
    'type_id': 'dmcs',
    'kwargs': {'render_mode': 'rgb_array', 
               'max_episode_steps': 1000},
    'coverage_idx': np.array([0,1]),
    'render_settings': {'x_lim': [-0.5, 0.5], 'y_lim': [-0.5, 0.5]}
},

##########################       MUJOCO       ############################################

"HalfCheetah-v3": {
    'type_id': 'mujoco',
    'kwargs': {'render_mode': 'rgb_array', 
               'reset_noise_scale': 0.0, 
               'exclude_current_positions_from_observation': False, 
               'max_episode_steps': 1000},
    'coverage_idx': np.array([0,1]),
    'render_settings': {'x_lim': [-50, 50], 'y_lim': [-1, 1]}
},
"Hopper-v3": {
    'type_id': 'mujoco',
    'kwargs': {'render_mode': 'rgb_array', 
               'reset_noise_scale': 0.0, 
               'exclude_current_positions_from_observation': False, 
               'max_episode_steps': 1000, 
               'terminate_when_unhealthy': True},
    'coverage_idx': np.array([0,1]),
    'render_settings': {'x_lim': [-40,40], 'y_lim': [0, 2]}
},
"Ant-v3": {
    'type_id': 'mujoco',
    'kwargs': {'render_mode': 'rgb_array', 
               'reset_noise_scale': 0.0, 
               'exclude_current_positions_from_observation': False, 
               'max_episode_steps': 1000,
               'terminate_when_unhealthy': False},
    'coverage_idx': np.array([0,1,2]),
    'render_settings': {'x_lim': [-50, 50], 'y_lim': [-50, 50], 'z_lim': [0, 2]}
},
"Walker2d-v3": {
    'type_id': 'mujoco',
    'kwargs': {'render_mode': 'rgb_array', 
               'reset_noise_scale': 0.0, 
               'exclude_current_positions_from_observation': False, 
               'max_episode_steps': 1000,
               'terminate_when_unhealthy': False},
    'coverage_idx': np.array([0,1]),
    'render_settings': {'x_lim': [-40, 40], 'y_lim': [0, 2]}
},
"Humanoid-v3": {
    'type_id': 'mujoco',
    'kwargs': {'render_mode': 'rgb_array', 
               'reset_noise_scale': 0.0, 
               'exclude_current_positions_from_observation': False, 
               'max_episode_steps': 1000,
               'terminate_when_unhealthy': False},
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


##########################       ATARI       ############################################
# MontezumaRevengeNoFrameskip-v4
"MontezumaRevengeNoFrameskip-v4": {
    'type_id': 'atari',
    'kwargs': {'frameskip': 4, 
               'max_episode_steps': 10000, 
               'render_mode': 'rgb_array'},
    'coverage_idx': np.array([0,1]),
    'render_settings': {'x_lim': [-1, 1], 'y_lim': [-1, 1]}
},
# PitfallNoFrameskip-v4
"PitfallNoFrameskip-v4": {
    'type_id': 'atari',
    'kwargs': {'frameskip': 4,
               'max_episode_steps': 10000, 
               'render_mode': 'rgb_array'},
    'coverage_idx': np.array([0,1]),
    'render_settings': {'x_lim': [-1, 1], 'y_lim': [-1, 1]}
},

}