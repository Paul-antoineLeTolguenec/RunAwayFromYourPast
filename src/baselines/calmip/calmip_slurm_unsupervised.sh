#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --begin=now
#SBATCH --mail-user=paul-antoine.le-tolguenec@isae.fr
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=slurm_calmip_unsupervised
#SBATCH --output=slurm_calmip_unsupervised_%j.out
#SBATCH --error=slurm_calmip_unsupervised_%j.err
#SBATCH --export=ALL


# FIND + RUN 
algo=${1:-../apt_ppo.py}
algo_id=$(basename "$algo" | sed 's/\.py//')




# MODULES

# Get the path to the config file
CONFIG_FILE="../../../envs/config_env.py"
HYPERPARAMETERS_FILE="../hyper_parameters.json"
# Extract env_ids from config file
env_ids=$(grep -oP '(?<=^")[^"]+(?=":)' "$CONFIG_FILE")
# FUNCTION: extract_hyperparameters
EXTRACT_SCRIPT="extract_hyperparameters.py"


# WANDB MODE
WANDB_MODE_ARG=${2:-"offline"}
if [ "$WANDB_MODE_ARG" == "offline" ]; then
    export WANDB_MODE="dryrun"

# CHECK IF WANDB MODE HAS BEEN SET
echo "WANDB_MODE: $WANDB_MODE"

# WANDB DIR 
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *"pando"* ]]; then
    export WANDB_DIR="/scratch/disc/p.le-tolguenec/"
elif [[ "$HOSTNAME" == *"olympe"* ]]; then
    export WANDB_DIR="/tmpdir/$USER/"

# CHECK IF WANDB_DIR HAS BEEN SET
echo "WANDB_DIR: $WANDB_DIR"

EXPERIMENT_NAME="${algo_id}_experiment"
SBATCH_JOB_NAME="slurm_${EXPERIMENT_NAME}"
SBATCH_OUT="slurm_${EXPERIMENT_NAME}_%j.out"
SBATCH_ERR="slurm_${EXPERIMENT_NAME}_%j.err"

# COUNT
execution_count=0


for env_id in $env_ids; do
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
            # srun --exclusive -N1 -n1 -c4 $cmd &
            ((execution_count++))
        done
    else
        echo "Skipping $env_id as it is of type 'atari'"
    fi
done
wait 
echo "Number of Python files executed: $execution_count"
