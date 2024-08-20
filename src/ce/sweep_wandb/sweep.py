import subprocess
import concurrent.futures
from functools import partial
import wandb
import tyro 
from dataclasses import dataclass
import os
import json
from src.ce.calmip.extract_hyperparameters import recursive_build_hyperparameters
import numpy as np

@dataclass
class Args:
    slurm_mode : bool = True
    """ calling srun instead of python  """
    wandb_project_name: str = "run_away_sweep"
    """the wandb project to log to for the sweeps"""
    algo : str = "v1klsac"
    """the algorithm to run"""
    make_gif: bool = False
    """whether to make a gif of the final policy"""
    type_id : str = "maze"
    """type of the environment to run"""
    nb_seeds : int = 3
    """number of seeds to run"""
    metric_to_maximize : str = "charts/coverage"
    """the metric to maximize"""
    nb_count : int = 5


    """number of runs to launch"""
    total_timesteps: int = 2_000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""


# ENV TO RUN PER TYPE
ENV_TYPE = {
    'maze': 'Maze-Ur-v0',
    'robotics': 'FetchReach-v1',
    'mujoco': 'Hoppe-v3'
    }


def run_script(file_algo, hp_cmd, seed, env_id, slurm):
    # if slurm : 
    #     cmd = f"srun -n1 -c7 poetry run python {file_algo} {hp_cmd} --seed {seed} --env_id {env_id}"
    # else : 
    cmd = f"poetry run python {file_algo} {hp_cmd} --seed {seed} --env_id {env_id}"
    print(cmd)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    last_line = result.stdout.strip().split('\n')[-1]
    return last_line

def train(args : dict[str], 
          algo : str, 
          nb_seeds : int, 
          type_id : str, 
          env_id : str,
          metric_to_maximize : str,
          slurm : bool,
          wandb_config : dict[str]) -> float:
    # set seeds
    seeds = list(range(nb_seeds))
    # replace args with wandb_config
    args.update(wandb_config)
    # update track 
    args['track'] = False
    # run the training script with the given hyperparameters
    hp_cmd = recursive_build_hyperparameters(args)
    file_algo = f"../{algo}.py"
    # run the training script with the given hyperparameters
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_script, file_algo, hp_cmd, seed, env_id, slurm) for seed in seeds]
        concurrent.futures.wait(futures)
        results = np.array([[float(f.split('=')[-1]) for f in future.result().split(',')] for future in futures]).mean(axis=0)
        coverage = results[0]
        shannon_entropy = results[1]
        episodic_return = results[2]
        return coverage, shannon_entropy, episodic_return
    
    
if __name__ == "__main__":
    args = tyro.cli(Args)
    algo = args.algo

    # extract hyperparameters
    with open("../hyper_parameters.json", 'r') as f:
        hyperparams = json.load(f)['hyperparameters'][args.type_id][algo]

    # replace hp : wandb project name + make_gif 
    hyperparams['wandb_project_name'] = args.wandb_project_name
    hyperparams['make_gif'] = args.make_gif
    hyperparams['total_timesteps'] = args.total_timesteps
    hyperparams['buffer_size'] = args.buffer_size

    # build main objective
    objective = partial(train, 
                   args=hyperparams,
                   algo=algo, 
                   nb_seeds=args.nb_seeds, 
                   type_id=args.type_id, 
                   env_id=ENV_TYPE[args.type_id],
                   metric_to_maximize=args.metric_to_maximize,
                   slurm = args.slurm_mode)

                   

    # main function
    def main():
        # log hyperparameters
        wandb.init(project=hyperparams['wandb_project_name'])
        # train
        coverage, shannon_entropy, episodic_return = objective(wandb_config=wandb.config)
        # log metric
        wandb.log({"charts/coverage": coverage, 
                   "charts/shannon_entropy": shannon_entropy, 
                   "charts/episodic_return": episodic_return})
    
    ######################################################### BEGIN SWEEP ##########################################################
    # SWEEP CONFIG
    if args.algo == 'v1klsac':
        sweep_params ={
                        "beta_ratio" : {"min": 1/512, "max": 1/4},
                        "lr_classifier" : {"min": 1e-4, "max": 1e-3},
                        "alpha" : {"min": 0.01, "max": 0.2},
                        "nb_episodes_rho" : {"values": [2, 4, 8, 16]},
                        "beta_noise" : {"min": 0.0, "max": 2.0},
                        }
    elif args.algo == 'v1wsac':
        sweep_params ={
                        "beta_ratio" : {"min": 1/512, "max": 1/4},
                        "lr_discriminator" : {"min": 1e-4, "max": 1e-3},
                        "alpha" : {"min": 0.01, "max": 0.2},
                        "nb_episodes_rho" : {"values": [2, 4, 8, 16]},
                        "beta_noise" : {"min": 0.0, "max": 2.0},
                        }
    elif args.algo == 'v2klsac':
        sweep_params ={
                        "beta_ratio" : {"min": 1/512, "max": 1/4},
                        "lr_classifier" : {"min": 1e-4, "max": 1e-3},
                        "alpha" : {"min": 0.01, "max": 0.2},
                        "nb_episodes_rho" : {"values": [2, 4, 8, 16]},
                        "beta_noise" : {"min": 0.0, "max": 2.0},
                        "lambda_diayn" : {"min": 0.0, "max": 1.0},
                        "lambda_kl" : {"min": 0.0, "max": 1.0},
                        }
    elif args.algo == 'v2wsac':
        sweep_params ={
                        "beta_ratio" : {"min": 1/512, "max": 1/4},
                        "lr_discriminator" : {"min": 1e-4, "max": 1e-3},
                        "alpha" : {"min": 0.01, "max": 0.2},
                        "nb_episodes_rho" : {"values": [2, 4, 8, 16]},
                        "beta_noise" : {"min": 0.0, "max": 2.0},
                        "lambda_reward_metra" : {"min": 0.0, "max": 1.0},
                        "lambda_wasserstein" : {"min": 0.0, "max": 1.0},
                        }
    else : 
        raise ValueError("algo not implemented")

    sweep_config = {
        "name": f"{algo}_{args.type_id}_{ENV_TYPE[args.type_id]}",
        "method": "bayes",
        "metric": {"name": args.metric_to_maximize, "goal": "maximize"},
        "parameters": sweep_params,
        }
    # initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project_name)
    
    # launch sweep
    wandb.agent(sweep_id, function=main, count=args.nb_count)

    

