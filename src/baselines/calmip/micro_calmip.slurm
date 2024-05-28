#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --begin=now
#SBATCH --mail-user=paul-antoine.le-tolguenec@isae.fr
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=slurm_calmip_unsupervised
#SBATCH --export=ALL


# FIND + RUN 
algo=${1:-../apt_ppo.py}
algo_id=$(basename "$algo" | sed 's/\.py//')


# Get the path to the config file
CONFIG_FILE="../../../envs/config_env.py"
# Get the path to the config file
HYPERPARAMETERS_FILE="../hyper_parameters.json"
# Extract env_ids from config file
env_id=${2:-"Maze-Easy-v0"}
# FUNCTION: extract_hyperparameters
EXTRACT_SCRIPT="extract_hyperparameters.py"


# WANDB MODE
WANDB_MODE_ARG=${3:-"offline"}
if [ "$WANDB_MODE_ARG" == "offline" ]; then
    export WANDB_MODE="dryrun"
fi

# CHECK IF WANDB MODE HAS BEEN SET
echo "WANDB_MODE: $WANDB_MODE"

# WANDB DIR 
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *"pando"* ]]; then
    export WANDB_DIR="/scratch/disc/p.le-tolguenec/"
elif [[ "$HOSTNAME" == *"olympe"* ]]; then
    export WANDB_DIR="/tmpdir/$USER/"
fi


# COUNT
execution_count=0
# Extract type_id 
type_id=$(awk -v env_id="$env_id" '
    BEGIN { FS="[:,]" }
    $1 ~ env_id { found=1 }
    found && /type_id/ { gsub(/[ "\t]/, "", $2); print $2; exit }
' "$CONFIG_FILE")

# Extract hyperparameters
cmd_hyperparams="poetry run python \"$EXTRACT_SCRIPT\" \"$HYPERPARAMETERS_FILE\" $type_id \"$algo_id\""
hyperparams=$(eval $cmd_hyperparams)
# hyperparams=$(poetry run python "$EXTRACT_SCRIPT" "$HYPERPARAMETERS_FILE" "$type_id" "$algo" "$algo_id")

if [ "$type_id" != "'atari'" ]; then
    for seed in {1..5}; do
        cmd="poetry run python $algo --env_id $env_id $hyperparams --seed $seed"
        echo $cmd 
        # $cmd
        srun --exclusive -N1 -n1 -c4 $cmd &
        ((execution_count++))
    done
else
    echo "Skipping $env_id as it is of type 'atari'"
fi
wait 
echo "Number of Python files executed: $execution_count"
