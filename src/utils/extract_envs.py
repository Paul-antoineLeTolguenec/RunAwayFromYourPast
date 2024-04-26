import numpy as np  
from envs.config_env import config
from dataclasses import dataclass
import tyro

@dataclass
class Args:
    type_id: str
    """ The type of the environment. """

args = tyro.cli(Args)
for env_name, env_info in config.items():
    if env_info['type_id'] == args.type_id:
        print(env_name)
