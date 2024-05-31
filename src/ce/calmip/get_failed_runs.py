import wandb
import sys
import json

def get_failed_runs(project_name):
    # Connect to W&B
    api = wandb.Api()
    
    # Fetch all runs for the given project
    runs = api.runs(project_name)
    
    # Dictionary to store failed or crashed runs
    failed_runs = {}
    
    for run in runs:
        # Check if the run state is 'failed' or 'crashed'
        if run.state in ['failed', 'crashed']:
            run_info = {
                'id': run.id,
                'hyperparameters': run.config
            }
            failed_runs[run.name] = run_info
    
    return failed_runs

if __name__ == "__main__":
    project_name = sys.argv[1]
    failed_runs = get_failed_runs(project_name)
    print(json.dumps(failed_runs))
