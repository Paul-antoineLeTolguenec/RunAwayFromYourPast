
# FIND + RUN 
algo=${1:-../apt_ppo.py}
algo_id=$(basename "$algo" | sed 's/\.py//')


# Get the path to the config file
CONFIG_FILE="../../../envs/config_env.py"
# Extract env_ids from config file
env_ids=$(grep -oP '(?<=^")[^"]+(?=":)' "$CONFIG_FILE")

# initialize
count=0

# for each env_id : sbatch micro_calmip.slurm
for env_id in $env_ids; do
    # Extract type_id 
    type_id=$(awk -v env_id="$env_id" '
        BEGIN { FS="[:,]" }
        $1 ~ env_id { found=1 }
        found && /type_id/ { gsub(/[ "\t]/, "", $2); print $2; exit }
    ' "$CONFIG_FILE")

    if [ "$type_id" != "'atari'" ]; then
        cmd="bash micro_calmip.slurm $algo $env_id offline --output=$algo_id-$env_id.out --error=$algo_id-$env_id.err --job-name=$algo_id-$env_id"
        eval $cmd
    else
        echo "Skipping $env_id as it is of type 'atari'"
    fi

    # counter for test 
    # Incrémenter le compteur
    count=$((count + 1))

    # Vérifier si le compteur est supérieur à 2
    if [ $count -gt 0 ]; then
        break
    fi

done
