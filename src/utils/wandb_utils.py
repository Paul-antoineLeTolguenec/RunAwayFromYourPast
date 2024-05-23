import wandb
import numpy as np
import pandas as pd


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
    return obs, action, reward, next_obs, done, times

def find_run_id(project_name: str, run_name: str):
    api = wandb.Api()
    runs = api.runs(project_name)
    for run in runs:
        if run.name == run_name:
            return run.id
    return None
    
   

   
if __name__ == "__main__":
    wandb.login()

    project_name = "contrastive_exploration"
    run_id = "Maze-Ur__aux_ppo__0"

    run_id = find_run_id(project_name, run_id)
    print(run_id)
    # dataset
    obs, action, reward, next_obs, done, times = load_dataset(project_name, run_id, "dataset")
    print('obs:', obs.shape)
    print('action:', action.shape)
    print('reward:', reward.shape)
    print('next_obs:', next_obs.shape)
    print('done:', done.shape)
    print('times:', times.shape)
