#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=7
#SBATCH --time=10:00:00
#SBATCH --job-name=v1klsac-Maze-Ur-v0
#SBATCH --output=/tmpdir/letolgue/error_out/v1klsac-Maze-Ur-v0-%j.out
#SBATCH --error=/tmpdir/letolgue/error_out/v1klsac-Maze-Ur-v0-%j.err
#SBATCH --mail-user=paul-antoine.le-tolguenec@isae.fr
#SBATCH --mail-type=FAIL
#SBATCH --begin=now
#SBATCH --export=ALL

# module load cuda/9.1.85.3

# find port 
find_available_port() {
    local port=$1
    while netstat -tuln | grep -q ":$port"; do
        port=1
    done
    echo $port
}

# Get the path to the config file
CONFIG_FILE="../../../envs/config_env.py"
# Get the path to the config file
HYPERPARAMETERS_FILE="../hyper_parameters.json"
# FUNCTION: extract_hyperparameters
EXTRACT_SCRIPT="extract_hyperparameters.py"


# WANDB MODE
if [ "online" == "offline" ]; then
    export WANDB_MODE="dryrun"
fi

# CHECK IF WANDB MODE HAS BEEN SET
echo "WANDB_MODE: $WANDB_MODE"

# WANDB DIR 
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *"pando"* ]]; then
    export WANDB_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_CACHE_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_CONFIG_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_ARTIFACTS_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_RUN_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_DATA_DIR="/scratch/disc/p.le-tolguenec/"
elif [[ "$HOSTNAME" == *"olympe"* ]]; then
    export WANDB_DIR="/tmpdir/$USER/"
    export WANDB_CACHE_DIR="/tmpdir/$USER/"
    export WANDB_CONFIG_DIR="/tmpdir/$USER/"
    export WANDB_ARTIFACTS_DIR="/tmpdir/$USER/"
    export WANDB_RUN_DIR="/tmpdir/$USER/"
    export WANDB_DATA_DIR="/tmpdir/$USER/"
fi


# COUNT
execution_count=0
# Extract type_id 
type_id=$(awk -v env_id="Maze-Ur-v0" '
    BEGIN { FS="[:,]" }
    $1 ~ env_id { found=1 }
    found && /type_id/ { gsub(/[ "\t]/, "", $2); print $2; exit }
' "$CONFIG_FILE")

# Extract hyperparameters
cmd_hyperparams="poetry run python \"$EXTRACT_SCRIPT\" \"$HYPERPARAMETERS_FILE\" $type_id \"v1klsac\""
hyperparams=$(eval $cmd_hyperparams)

for seed in {0..4}; do
    cmd="poetry run python ../v1klsac.py --env_id Maze-Ur-v0 $hyperparams --seed $seed"
    echo $cmd 
    # $cmd
    srun -n1 -c7 proxychains4 $cmd &
done

wait 
echo "Number of Python files executed: $execution_count"
