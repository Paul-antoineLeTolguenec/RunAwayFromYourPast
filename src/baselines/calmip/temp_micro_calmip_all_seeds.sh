#!/bin/bash

show_help() {
    echo "Usage: $0 --algo <script_algo> --types <type_list> --seeds <seed_list>"
    echo ""
    echo "Arguments:"
    echo "  --algo    Chemin vers le script d'algorithme (par défaut : ../v1_ppo_kl_adaptive_sampling.py)"
    echo "  --env_id  Identifiant de l'environnement (par défaut : Maze-Easy-v0)"
    echo "  --hp_file File containing hyperparameters (default: hyper_parameters.json)"
    echo "  --wandb_mode Wandb mode (default: offline)"
    echo ""
    echo "Exemple: $0 --algo ../v1klsac.py --types \"[robotics, mujoco]\" --seeds \"[3,4]\" --wandb_mode offline"
}

algo="..apt_sac.py"
env_id="Maze-Easy-v0"
HYPER_PARAMETERS_FILE="../hyper_parameters_sac.json"
WANDB_MODE_ARG="offline"



# Affichage des paramètres obtenus
# Analyse des arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --algo)
            algo="$2"
            shift 2
            ;;
        --env_id)
            env_id="$2"
            shift 2
            ;;
        --hp_file)
            HYPER_PARAMETERS_FILE="$2"
            shift 2
            ;;
        --wandb_mode)
            WANDB_MODE_ARG="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Argument inconnu : $1"
            show_help
            exit 1
            ;;  
    esac
done

# Arguments passés au script

algo_id=$(basename "$algo" | sed 's/\.py//')

# WANDB MODE
if [ "$WANDB_MODE_ARG" == "offline" ]; then
    export WANDB_MODE="dryrun"
fi

# CHECK IF WANDB MODE HAS BEEN SET
echo "WANDB_MODE: $WANDB_MODE"

# WANDB DIR 
HOSTNAME=$(hostname)
USERNAME=$(whoami)
if [[ "$HOSTNAME" == *"pando"* ]]; then
    export WANDB_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_CACHE_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_CONFIG_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_ARTIFACTS_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_RUN_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_DATA_DIR="/scratch/disc/p.le-tolguenec/"
elif [[ "$HOSTNAME" == *"olympe"* ]]; then
    if [[ "$USER" != "p21049lp" && "$USER" != "letolgue" ]]; then
        export WANDB_DIR="/tmpdir/$USER/P_A/"
        export WANDB_CACHE_DIR="/tmpdir/$USER/P_A/"
        export WANDB_CONFIG_DIR="/tmpdir/$USER/P_A/"
        export WANDB_ARTIFACTS_DIR="/tmpdir/$USER/P_A/"
        export WANDB_RUN_DIR="/tmpdir/$USER/P_A/"
        export WANDB_DATA_DIR="/tmpdir/$USER/P_A/"
    else
        export WANDB_DIR="/tmpdir/$USER/"
        export WANDB_CACHE_DIR="/tmpdir/$USER/"
        export WANDB_CONFIG_DIR="/tmpdir/$USER/"
        export WANDB_ARTIFACTS_DIR="/tmpdir/$USER/"
        export WANDB_RUN_DIR="/tmpdir/$USER/"
        export WANDB_DATA_DIR="/tmpdir/$USER/"
    fi
fi

# create tempfile
temp_slurm_script="temp_slurm_script_$$.slurm"

# Check HOSTNAME
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *"pando"* ]]; then
    path_file_err_out="/scratch/disc/p.le-tolguenec/error_out/"
elif [[ "$HOSTNAME" == *"olympe"* ]]; then
    if [[ "$USER" != "p21049lp" && "$USER" != "letolgue" ]]; then
        path_file_err_out="/tmpdir/$USER/P_A/error_out/"
    else
        path_file_err_out="/tmpdir/$USER/error_out/"
    fi
else 
    path_file_err_out="/tmp/error_out/"
fi

# create error output directory if it does not exist
mkdir -p "$path_file_err_out"


cat <<EOT > $temp_slurm_script
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --time=120:00:00
#SBATCH --job-name=$algo_id-$env_id
#SBATCH --output=$path_file_err_out$algo_id-$env_id-%j.out
#SBATCH --error=$path_file_err_out$algo_id-$env_id-%j.err
#SBATCH --mail-user=paul-antoine.le-tolguenec@isae.fr
#SBATCH --mail-type=FAIL
#SBATCH --mem=170G          # Set a memory limit
#SBATCH --acctg-freq=task=1 # Set a memory check frequency in second (60s by default)
#SBATCH --begin=now
#SBATCH --export=ALL

# module load cuda/9.1.85.3
module load misc-libs/libOSMesa 
module load gcc

# find port 
find_available_port() {
    local port=\$1
    while netstat -tuln | grep -q ":\$port"; do
        port=$((port + 1))
    done
    echo \$port
}

# Get the path to the config file
CONFIG_FILE="../../../envs/config_env.py"
# Get the path to the config file
HYPERPARAMETERS_FILE="$HYPER_PARAMETERS_FILE"
# FUNCTION: extract_hyperparameters
EXTRACT_SCRIPT="extract_hyperparameters.py"




echo "WANDB_DIR: \$WANDB_DIR"
echo "WANDB_CACHE_DIR: \$WANDB_CACHE_DIR"
echo "WANDB_CONFIG_DIR: \$WANDB_CONFIG_DIR"
echo "WANDB_ARTIFACTS_DIR: \$WANDB_ARTIFACTS_DIR"
echo "WANDB_RUN_DIR: \$WANDB_RUN_DIR"
echo "WANDB_DATA_DIR: \$WANDB_DATA_DIR"



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

for seed in {0..4}; do
    cmd="poetry run python $algo --env_id $env_id \$hyperparams --seed \$seed"
    echo \$cmd 
    # \$cmd
    srun --exclusive -N1 -n1 \$cmd 
done
echo "Number of Python files executed: \$execution_count"
EOT

# Soumettre le script temporaire
sbatch $temp_slurm_script
# cat $temp_slurm_script

# Supprimer le fichier temporaire après soumission
rm $temp_slurm_script
