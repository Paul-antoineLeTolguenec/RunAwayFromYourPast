#!/bin/bash

# Vérification de l'hostname pour définir le chemin des dossiers offline
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *"pando"* ]]; then
    path_offline="/scratch/disc/p.le-tolguenec/wandb/"
elif [[ "$HOSTNAME" == *"olympe"* ]]; then
    path_offline="/tmpdir/$USER/wandb/"
else
    echo "Hostname non reconnu. Script annulé."
    exit 1
fi

# Vérifier si au moins un argument est fourni
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <algorithm_name1> <algorithm_name2> ..."
    exit 1
fi

ALGO_NAMES=("$@")

# Fonction pour extraire la valeur d'un argument spécifique dans wandb-metadata.json
extract_arg_value() {
    local json_file=$1
    local arg_name=$2
    grep -A1 "\"--$arg_name\"" "$json_file" | tail -n1 | awk -F'"' '{print $2}'
}

# Fonction pour vérifier si un fichier contient au moins un des algorithmes
contains_algo() {
    local json_file=$1
    for algo in "${ALGO_NAMES[@]}"; do
        if grep -q "$algo" "$json_file"; then
            return 0
        fi
    done
    return 1
}

count=0

# Parcourir les dossiers commençant par "offline" dans le répertoire défini
for DIR in "$path_offline"offline*; do
    # Vérifier si c'est un dossier
    if [ -d "$DIR" ]; then
        # Chemin vers le fichier wandb-metadata.json
        METADATA_FILE="$DIR/files/wandb-metadata.json"
        
        # Vérifier si le fichier existe
        if [ -f "$METADATA_FILE" ]; then
            # Vérifier si le fichier contient au moins un des algorithmes
            if contains_algo "$METADATA_FILE"; then
                # Extraire l'env_id
                ENV_ID=$(extract_arg_value "$METADATA_FILE" "env_id")
                count=$((count + 1))
                echo "Synchronizing $DIR with env_id: $ENV_ID..."
                wandb sync "$DIR"
                if [ $? -eq 0 ]; then
                    echo "Successfully synchronized $DIR with env_id: $ENV_ID"
                else
                    echo "Failed to synchronize $DIR with env_id: $ENV_ID"
                fi
            fi
        fi
    fi
done

echo "Total $count directories synchronized"
