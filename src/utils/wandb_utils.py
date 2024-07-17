import wandb
import numpy as np
import pandas as pd
import re


def send_video(wandb: wandb, video: np.ndarray, name: str, step: int):
    print('video shape:', video.shape)  
    wandb.log({name: wandb.Video(video, fps=4, format="gif")}, step=step)

def send_matrix(wandb: wandb, matrix: np.ndarray, name: str, step: int):
    wandb.log({name: wandb.Image(matrix)}, step=step)

def send_dataset(wandb: wandb, obs: np.ndarray,
                 action: np.ndarray, reward: np.ndarray,
                    next_obs: np.ndarray, done: np.ndarray,
                    times : np.ndarray, name: str, step: int):
    df_obs = pd.DataFrame(obs, columns=[f"obs_{i}" for i in range(obs.shape[1])])
    df_action = pd.DataFrame(action, columns=[f"action_{i}" for i in range(action.shape[1])])
    df_reward = pd.DataFrame(reward, columns=["reward"])
    df_next_obs = pd.DataFrame(next_obs, columns=[f"next_obs_{i}" for i in range(next_obs.shape[1])])
    df_done = pd.DataFrame(done, columns=["done"])
    df_times = pd.DataFrame(times, columns=["times"])
    df = pd.concat([df_obs, df_action, df_reward, df_next_obs, df_done, df_times], axis=1)
    wandb.log({name: wandb.Table(dataframe=df)}, step=step)

def load_dataset(project_name: str, run_id: str, dataset_name: str):
    artifact = wandb.Api().artifact(f"{project_name}/run-{run_id}-{dataset_name}:v0")
    table = artifact.get(dataset_name)
    df = table.get_dataframe()
    obs_col = [col for col in df.columns if col.startswith("obs")]
    obs = df[obs_col].values
    action_col = [col for col in df.columns if col.startswith("action")]
    action = df[action_col].values
    reward = df["reward"].values
    next_obs_col = [col for col in df.columns if col.startswith("next_obs")]
    next_obs = df[next_obs_col].values
    done = df["done"].values
    times = df["times"].values
    return obs, action, np.expand_dims(reward, axis=1), next_obs, np.expand_dims(done, axis=1), np.expand_dims(times, axis=1)

def find_run_id(project_name: str, run_name: str):
    api = wandb.Api()
    runs = api.runs(project_name)
    for run in runs:
        if run.name == run_name:
            return run.id
    return None

def get_failed_runs(project_name: str, status: list = ["failed", "crashed"], algo_name: str = None, remove: bool = False):
    api = wandb.Api()
    runs = api.runs(project_name)
    failed_runs = []
    for run in runs:
        if run.state == "failed" or run.state == "crashed":
            if algo_name is None:
                failed_runs.append(run.name)
            else:
                if algo_name in run.name:
                    failed_runs.append(run.name)
                else:
                    None
    if remove:
        for run_name in failed_runs:
            run_id = find_run_id(project_name, run_name)
            api.run(f"{project_name}/{run_id}").delete()
    
    return failed_runs
    
def sanitize_name(name):
    """
    Sanitize the artifact name to contain only allowed characters.
    """
    return re.sub(r'[^a-zA-Z0-9_\-.]', '_', name)

def run_exists_in_target_project(api, target_project, run_id):
    """
    Check if a run with the given ID exists in the target project.
    
    Args:
        api (wandb.Api): The wandb API client.
        target_project (str): The name of the target project.
        run_id (str): The ID of the run to check.
        
    Returns:
        bool: True if the run exists in the target project, False otherwise.
    """
    target_runs = api.runs(target_project)
    for run in target_runs:
        if run.id == run_id:
            return True
    return False

def transfer_finished_runs(source_project, target_project):
    """
    Transfers finished runs from one Weights & Biases (wandb) project to another.
    
    Args:
        source_project (str): The name of the source project from which to fetch runs.
        target_project (str): The name of the target project to which finished runs will be uploaded.
    """
    # Initialize a new wandb API client
    api = wandb.Api()
    
    # Fetch all runs from the source project
    source_runs = api.runs(source_project)
    nb_transfered_runs = 0
    for run in source_runs:
        # Check if the run status is 'finished'
        if run.state == 'finished':
            # Check if the run already exists in the target project
            if run_exists_in_target_project(api, target_project, run.id):
                print(f"Run {run.id} already exists in {target_project}, skipping transfer.")
                continue
            
            print(f"Transferring run {run.id} from {source_project} to {target_project}")
            
            # Initialize a new run in the target project
            with wandb.init(project=target_project, reinit=True):
                # Transfer run metadata
                wandb.config.update(run.config)
                
                # Transfer metrics and other information
                history = run.history(pandas=False)
                for entry in history:
                    wandb.log(entry)
                
                # Transfer artifacts if any
                # for artifact in run.logged_artifacts():
                #     artifact.download(root="artifacts")
                #     sanitized_name = sanitize_name(artifact.name)
                #     new_artifact = wandb.Artifact(sanitized_name, type=artifact.type)
                #     new_artifact.add_dir("artifacts")
                #     wandb.log_artifact(new_artifact)
                
                # Finish the run
                wandb.finish()
            
            print(f"Run {run.id} transferred successfully.")

    print(f"Transferred {nb_transfered_runs} runs from {source_project} to {target_project}.")
   
if __name__ == "__main__":
    # project_name = "contrastive_exploration"
    # failed_runs = get_failed_runs(project_name, algo_name='ngu', remove=True)
    # print(failed_runs)
    # print('nb failed runs:', len(failed_runs))
    # # dataset
    # print('project_name:', project_name)
    # print('run_id:', run_id)
    # print('dataset:', "dataset")
    # obs, action, reward, next_obs, done, times = load_dataset(project_name, run_id, "dataset")
    # print('obs:', obs.shape)
    # print('action:', action.shape)
    # print('reward:', reward.shape)
    # print('next_obs:', next_obs.shape)
    # print('done:', done.shape)
    # print('times:', times.shape)

    # check get failed runs
    source_project_name = "contrastive_test"
    target_project_name = "contrastive_exploration_reward_max"

    transfer_finished_runs(source_project_name, target_project_name)