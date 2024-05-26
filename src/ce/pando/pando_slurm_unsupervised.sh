
# Get the path to the config file
CONFIG_FILE="../../../envs/config_env.py"
HYPERPARAMETERS_FILE="../hyper_parameters.json"
# Extract env_ids from config file
env_ids=$(grep -oP '(?<=^")[^"]+(?=":)' "$CONFIG_FILE")

# FUNCTION: extract_hyperparameters
EXTRACT_SCRIPT="extract_hyperparameters.py"


# FIND + RUN 
algo=${1:-../v1_ppo_kl_adaptive_sampling.py}

for env_id in $env_ids; do
    # Extract type_id 
    type_id=$(awk -v env_id="$env_id" '
        BEGIN { FS="[:,]" }
        $1 ~ env_id { found=1 }
        found && /type_id/ { gsub(/[ "\t]/, "", $2); print $2; exit }
    ' "$CONFIG_FILE")

    # Extract hyperparameters
    hyperparams=$(poetry run python "$EXTRACT_SCRIPT" "$HYPERPARAMETERS_FILE" "$type_id")

    echo $hyperparams

    if [ "$type_id" != "'atari'" ]; then
        cmd="poetry run python \"$algo\" --env_id \"$env_id\""
        echo $cmd
        # $cmd
    else
        echo "Skipping $env_id as it is of type 'atari'"
    fi
done
