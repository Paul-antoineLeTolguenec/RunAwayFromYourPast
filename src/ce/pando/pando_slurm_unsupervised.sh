
# Get the path to the config file
CONFIG_FILE="../../../envs/config_env.py"
HYPERPARAMETERS_FILE="../hyper_parameters.json"
# Extract env_ids from config file
env_ids=$(grep -oP '(?<=^")[^"]+(?=":)' "$CONFIG_FILE")

# FUNCTION: extract_hyperparameters
extract_hyperparameters() {
    local type_id=$1
    local hyperparams=""
    local in_type_id_section=0

    while IFS= read -r line; do
        # Check if we are entering the relevant section
        if [[ $line =~ \"$type_id\" ]]; then
            in_type_id_section=1
        elif [[ $in_type_id_section -eq 1 && $line =~ \} ]]; then
            # End of the current type_id section
            in_type_id_section=0
        fi
        echo "in_type_id_section: $in_type_id_section"
        # If we are in the relevant section, process the line
        if [[ $in_type_id_section -eq 1 && $line =~ : ]]; then
            key=$(echo "$line" | grep -oP '\"[^\"]+\"' | head -1 | tr -d '\"')
            value=$(echo "$line" | grep -oP ':.*' | cut -d: -f2 | tr -d ' ",')
            if [[ "$value" == "true" ]]; then
                value="True"
            elif [[ "$value" == "false" ]]; then
                value="False"
            fi
            if [[ -n "$key" && -n "$value" ]]; then
                hyperparams="$hyperparams --$key $value"
            fi
        fi
    done < "$HYPERPARAMETERS_FILE"
    
    echo "$hyperparams"
}

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
    hyperparams=$(extract_hyperparameters $type_id)

    echo $hyperparams

    if [ "$type_id" != "'atari'" ]; then
        cmd="poetry run python \"$algo\" --env_id \"$env_id\""
        echo $cmd
        # $cmd
    else
        echo "Skipping $env_id as it is of type 'atari'"
    fi
done
