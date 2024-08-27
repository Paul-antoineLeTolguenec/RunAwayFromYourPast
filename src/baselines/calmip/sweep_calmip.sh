
# Function to display help message if the script is misused
show_help() {
    echo "Usage: sbatch fichier.slurm --algo ALGO_NAME --type_id TYPE_ID"
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
    echo "Please ensure that both arguments are provided."
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
if [ -z "$algo" ] || [ -z "$type_id" ]|| [ -z "$WANDB_MODE" ]; then
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

# create error output directory if it does not exist
mkdir -p "$path_file_err_out"

# create temporary directory if it does not exist
tmp_dir="/tmpdir/$USER/tmp"
mkdir -p "$tmp_dir"

# create tempfile
temp_slurm_script="temp_slurm_script.slurm"

cat <<EOT > $temp_slurm_script
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=7
#SBATCH --time=10:00:00
#SBATCH --job-name=sweep-$algo-$type_id
#SBATCH --output=$path_file_err_out$algo-$type_id-%j.out
#SBATCH --error=$path_file_err_out$algo-$type_id-%j.err
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


cmd="proxychains4 poetry run python ../sweep_wandb/sweep.py --algo $algo --type_id $type_id"
echo \$cmd 
eval \$cmd
# srun proxychains4 \$cmd 

EOT

# Soumettre le script temporaire
sbatch $temp_slurm_script
# cat $temp_slurm_script

# Supprimer le fichier temporaire après soumission
rm $temp_slurm_script
