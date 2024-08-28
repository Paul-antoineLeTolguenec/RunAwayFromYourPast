#!/bin/bash

# Function to display help message if the script is misused
show_help() {
    echo "Usage: sbatch fichier.slurm --algo ALGO_NAME --type_id TYPE_ID --wandb_mode WANDB_MODE"
    echo
    echo "This script submits a SLURM job with specific parameters."
    echo
    echo "Arguments:"
    echo "  --algo ALGO_NAME    Name of the algorithm to use"
    echo "  --type_id TYPE_ID   Identifier for the type"
    echo "  --wandb_mode WANDB_MODE set mode wandb"
    echo
    echo "Example:"
    echo "  sbatch fichier.slurm --algo v1klsac --type_id maze --wandb_mode online"
    echo
    echo "Please ensure that all arguments are provided."
}

# Extracting the command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --algo) algo="$2"; shift ;;
        --type_id) type_id="$2"; shift ;;
        --wandb_mode) WANDB_MODE="$2"; shift ;;
        *) echo "Unknown argument: $1"; show_help; exit 1 ;;
    esac
    shift
done

# Check if the variables are set, if not, show the help message and exit
if [ -z "$algo" ] || [ -z "$type_id" ] || [ -z "$WANDB_MODE" ]; then
    echo "Error: Missing required arguments."
    show_help
    exit 1
fi

# Check HOSTNAME
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *"pando"* ]]; then
    path_file_err_out="/scratch/disc/p.le-tolguenec/error_out/"
elif [[ "$HOSTNAME" == *"olympe"* ]]; then
    path_file_err_out="/tmpdir/$USER/error_out/"
else 
    path_file_err_out="/tmp/error_out/"
fi


# Create tempfile slurm script
algo_id=$(basename "$algo" | sed 's/\.py//')
temp_slurm_script="temp_slurm_script.slurm"

# Determine env_id given the type_id
if [[ "$type_id" == "maze" ]]; then
    env_id="Maze-Ur-v0"
elif [[ "$type_id" == "robotics" ]]; then
    env_id="FetchReach-v1"
elif [[ "$type_id" == "mujoco" ]]; then
    env_id="Hopper-v3"
else
    echo "Unknown type_id: $type_id"
    exit 1
fi

echo "env_id: $env_id"

# Determine hyperparameters based on the algorithm
if [[ "$algo_id" == "v1klsac" ]]; then
    hyperparameters="
  beta_ratio:
    min: 0.001953125
    max: 0.25
  lr_classifier:
    min: 0.0001
    max: 0.001
  alpha:
    min: 0.01
    max: 0.2
  nb_episodes_rho:
    values: [2, 4, 8, 16]
  beta_noise:
    min: 0.0
    max: 2.0
  seed:
    values: [0, 1, 2, 3, 4]
    "
elif [[ "$algo_id" == "v1wsac" ]]; then
    hyperparameters="
  beta_ratio:
    min: 0.001953125
    max: 0.25
  lr_discriminator:
    min: 0.0001
    max: 0.001
  alpha:
    min: 0.01
    max: 0.2
  nb_episodes_rho:
    values: [2, 4, 8, 16]
  beta_noise:
    min: 0.0
    max: 2.0
  seed:
    values: [0, 1, 2, 3, 4]
    "
elif [[ "$algo_id" == "v2klsac" ]]; then
    hyperparameters="
  beta_ratio:
    min: 0.001953125
    max: 0.25
  lr_classifier:
    min: 0.0001
    max: 0.001
  alpha:
    min: 0.01
    max: 0.2
  nb_episodes_rho:
    values: [2, 4, 8, 16]
  beta_noise:
    min: 0.0
    max: 2.0
  lambda_diayn:
    min: 0.0
    max: 1.0
  lambda_kl:
    min: 0.0
    max: 1.0
  seed:
    values: [0, 1, 2, 3, 4]
    "
elif [[ "$algo_id" == "v2wsac" ]]; then
    hyperparameters="
  beta_ratio:
    min: 0.001953125
    max: 0.25
  lr_discriminator:
    min: 0.0001
    max: 0.001
  alpha:
    min: 0.01
    max: 0.2
  nb_episodes_rho:
    values: [2, 4, 8, 16]
  beta_noise:
    min: 0.0
    max: 2.0
  lambda_reward_metra:
    min: 0.0
    max: 1.0
  lambda_wasserstein:
    min: 0.0
    max: 1.0
  seed:
    values: [0, 1, 2, 3, 4]
    "
else
    echo "Unknown algorithm: $algo"
    exit 1
fi

# Générer le fichier YAML temporaire pour le sweep
sweep_yaml="temp_config.yaml"
cat <<EOF > $sweep_yaml
project: "run_away_sweep"
name: "$algo_id-sweep-$type_id"

method: bayes  
metric:
  name: charts/coverage
  goal: maximize

parameters:$hyperparameters

command:
  - poetry
  - run
  - python
  - $algo
  - --use_hp_file
  - --sweep_mode
  - --hp_file
  - ../hyper_parameters.json
  - --env_id
  - $env_id
EOF


# show sweep.yaml 
# cat $sweep_yaml

cat <<EOT > $temp_slurm_script
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=10:00:00
#SBATCH --job-name=sweep-$algo-$type_id
#SBATCH --output=$path_file_err_out$algo_id-$type_id-%j.out
#SBATCH --error=$path_file_err_out$algo_id-$type_id-%j.err
#SBATCH --mail-user=paul-antoine.le-tolguenec@isae.fr
#SBATCH --mail-type=FAIL
#SBATCH --mem=170G          # Set a memory limit
#SBATCH --acctg-freq=task=1 # Set a memory check frequency in second (60s by default)
#SBATCH --begin=now
#SBATCH --export=ALL


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




sweep_command="poetry run wandb sweep $sweep_yaml"
sweep_output=\$(\$sweep_command 2>&1)
sweep_id_cmd=\$(echo "\$sweep_output" | grep "wandb: Run sweep agent with:" | sed "s/.*wandb: Run sweep agent with: //")
echo "Sweep ID: \$sweep_id_cmd"



for seed in {0..1}; do
    cmd="poetry run \$sweep_id_cmd"
    echo \$cmd
    \$cmd &
    sleep 1
done


wait



EOT



cat $temp_slurm_script

# Soumettre le script temporaire
sbatch $temp_slurm_script
# bash $temp_slurm_script


# Supprimer le fichier temporaire après soumission
rm $temp_slurm_script


# delete config 
# rm $sweep_yaml