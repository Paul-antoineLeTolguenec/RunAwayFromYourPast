#!/bin/bash

#!/bin/bash

# Fonction pour afficher l'aide
show_help() {
    echo "Usage: $0 --algo <script_algo> --env <environment_id> [--mode <wandb_mode>] [--seed <seed_value>]"
    echo ""
    echo "Arguments:"
    echo "  --algo   Le chemin vers le script d'algorithme (par défaut : ../v1_ppo_kl_adaptive_sampling.py)"
    echo "  --env    L'ID de l'environnement (par défaut : Maze-Easy-v0)"
    echo "  --mode   Mode WANDB (par défaut : offline)"
    echo "  --seed   Valeur de la graine aléatoire (par défaut : non défini)"
    echo "  --hp_file   Hyperparameters file (par défaut : hyper_parameters_exploit.json)"
    echo ""
    echo "Exemple: $0 --algo ../v1_ppo_kl_adaptive_sampling.py --env Maze-Easy-v0 --mode online --seed 42 --hp_file hyper_parameters.json"
}

# Variables par défaut
algo="../v1_ppo_kl_adaptive_sampling.py"
env_id="Maze-Easy-v0"
WANDB_MODE_ARG="offline"
seed="0" # Seed par défaut vide
HYPERPARAMETERS_FILE="../hyper_parameters_exploit.json"

# Analyse des arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --algo)
            algo="$2"
            shift 2
            ;;
        --env)
            env_id="$2"
            shift 2
            ;;
        --mode)
            WANDB_MODE_ARG="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --hp_file)
            HYPERPARAMETERS_FILE="$2"
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

# Exécution principale avec les variables obtenues
algo_id=$(basename "$algo" | sed 's/\.py//')

echo "Exécution avec les paramètres suivants :"
echo "  Algo: $algo"
echo "  Algo ID: $algo_id"
echo "  Environment ID: $env_id"
echo "  WANDB Mode: $WANDB_MODE_ARG"
echo "  Seed: $seed"
echo "  Hyperparameters file: $HYPERPARAMETERS_FILE"


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
    export WANDB_DIR="/scratch/disc/p.le-tolguenec/online_wandb"
    export WANDB_CACHE_DIR="/scratch/disc/p.le-tolguenec/online_wandb"
    export WANDB_CONFIG_DIR="/scratch/disc/p.le-tolguenec/online_wandb"
    export WANDB_ARTIFACTS_DIR="/scratch/disc/p.le-tolguenec/online_wandb"
    export WANDB_RUN_DIR="/scratch/disc/p.le-tolguenec/online_wandb"
    export WANDB_DATA_DIR="/scratch/disc/p.le-tolguenec/online_wandb"
    # create folder if it does not exist
    mkdir -p "$WANDB_DIR"
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
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
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
HYPERPARAMETERS_FILE=$HYPERPARAMETERS_FILE
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

cmd="poetry run python $algo --env_id $env_id \$hyperparams --seed $seed"
echo \$cmd 
# \$cmd
if [[ "\$HOSTNAME" == *"pando"* ]]; then
srun --exclusive -N1 -n1 \$cmd 
elif [[ "\$HOSTNAME" == *"olympe"* ]]; then
proxychains4 srun --exclusive -N1 -n1 \$cmd 
fi

EOT

# Soumettre le script temporaire
sbatch $temp_slurm_script
# cat $temp_slurm_script

# Supprimer le fichier temporaire après soumission
rm $temp_slurm_script
