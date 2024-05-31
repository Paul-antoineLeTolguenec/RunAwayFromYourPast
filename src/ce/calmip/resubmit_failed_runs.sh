#!/bin/bash

# Arguments passés au script
project_name=${1:-"contrastive_exploration"}
algo_dir="../"
echo $algo_dir
# Répertoire courant contenant les scripts
script_dir=$(dirname "$0")
# create tempfile
temp_slurm_script="temp_slurm_script_$$.slurm"

# Check HOSTNAME
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *"pando"* ]]; then
    path_file_err_out="/scratch/disc/p.le-tolguenec/error_out/"
elif [[ "$HOSTNAME" == *"olympe"* ]]; then
    path_file_err_out="/tmpdir/$USER/error_out/"
else 
    path_file_err_out="/tmp/error_out/"
fi


# Fonction Python pour récupérer les runs échoués
get_failed_runs() {
    poetry run python "$script_dir/get_failed_runs.py" "$project_name"
}

# Récupérer les runs échoués
failed_runs=$(get_failed_runs)

# Parser les runs échoués
parse_failed_runs() {
    echo "$failed_runs" | poetry run python -c "
import sys, json
failed_runs = json.load(sys.stdin)
for run_name, run_info in failed_runs.items():
    algo = run_info['hyperparameters'].get('algo', 'v1_ppo_kl_adaptive_sampling.py')
    algo_id = algo.replace('.py', '')
    env_id = run_info['hyperparameters'].get('env_id', 'Maze-Easy-v0')
    seed = run_info['hyperparameters'].get('seed', 1)
    exp_name = run_info['hyperparameters'].get('exp_name', 'default_exp')
    print(f'{algo},{algo_id},{env_id},{seed},{exp_name}')
"
}

# Lire les runs échoués
runs_to_relaunch=$(parse_failed_runs)
# echo "Runs to relaunch: $runs_to_relaunch"

count = 0   

# Créer un script SLURM pour chaque run échoué
while IFS=',' read -r algo algo_id env_id seed exp_name; do
    algo_path="$algo_dir/$algo"
    echo "algo: $algo, env_id: $env_id, seed: $seed "
    if [ ! -f "$algo_path" ]; then
        echo "Algo script $algo_path not found, skipping..."
        continue
    fi

    cat <<EOT > $temp_slurm_script
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=7
#SBATCH --time=05:00:00
#SBATCH --job-name=run-$algo_id-$env_id
#SBATCH --output=$path_file_err_out$algo_id-$env_id-%j.out
#SBATCH --error=$path_file_err_out$algo_id-$env_id-%j.err
#SBATCH --mail-user=paul-antoine.le-tolguenec@isae.fr
#SBATCH --mail-type=FAIL
#SBATCH --begin=now
#SBATCH --export=ALL


# Get the path to the config file
CONFIG_FILE="../../../envs/config_env.py"
# Get the path to the config file
HYPERPARAMETERS_FILE="../hyper_parameters.json"
# FUNCTION: extract_hyperparameters
EXTRACT_SCRIPT="extract_hyperparameters.py"


# WANDB MODE
if [ "$WANDB_MODE_ARG" == "offline" ]; then
    export WANDB_MODE="dryrun"
fi

# CHECK IF WANDB MODE HAS BEEN SET
echo "WANDB_MODE: \$WANDB_MODE"

# WANDB DIR 
HOSTNAME=\$(hostname)
if [[ "\$HOSTNAME" == *"pando"* ]]; then
    export WANDB_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_CACHE_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_CONFIG_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_ARTIFACTS_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_RUN_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_DATA_DIR="/scratch/disc/p.le-tolguenec/"
elif [[ "\$HOSTNAME" == *"olympe"* ]]; then
    export WANDB_DIR="/tmpdir/\$USER/"
    export WANDB_CACHE_DIR="/tmpdir/\$USER/"
    export WANDB_CONFIG_DIR="/tmpdir/\$USER/"
    export WANDB_ARTIFACTS_DIR="/tmpdir/\$USER/"
    export WANDB_RUN_DIR="/tmpdir/\$USER/"
    export WANDB_DATA_DIR="/tmpdir/\$USER/"
fi


# COUNT
execution_count=0
# Extract type_id 
type_id=\$(awk -v env_id="$env_id" '
    BEGIN { FS="[:,]" }
    \$1 ~ env_id { found=1 }
    found && /type_id/ { gsub(/[ "\t]/, "", \$2); print \$2; exit }
' "\$CONFIG_FILE")

# Extract hyperparameters
cmd_hyperparams="poetry run python \"\$EXTRACT_SCRIPT\" \"\$HYPERPARAMETERS_FILE\" \$type_id \"$algo_id\""
hyperparams=\$(eval \$cmd_hyperparams)
# hyperparams=\$(poetry run python "\$EXTRACT_SCRIPT" "\$HYPERPARAMETERS_FILE" "\$type_id" "$algo" "$algo_id")

if [ "\$type_id" != "'atari'" ]; then
    cmd="poetry run python ../$algo \$hyperparams --seed $seed --env_id $env_id"
    echo \$cmd 
    srun --exclusive -N1 -n1 -c4 \$cmd &
    ((execution_count++))
else
    echo "Skipping $env_id as it is of type 'atari'"
fi
wait 
echo "Number of Python files executed: \$execution_count"
EOT

# Soumettre le script temporaire
sbatch $temp_slurm_script

count=$((count+1))

# cat $temp_slurm_script if count == 1

# Supprimer le fichier temporaire après soumission
rm $temp_slurm_script

done <<< "$runs_to_relaunch"

echo "Synchronization and re-launch process completed."
