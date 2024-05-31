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

# Vérifier si un argument est fourni
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <algorithm_name>"
    exit 1
fi

ALGO_NAME=$1

# Fonction pour extraire la valeur d'un argument spécifique dans wandb-metadata.json
extract_arg_value() {
    local json_file=$1
    local arg_name=$2
    jq -r ".args" "$json_file" | jq -r 'map(select(.[0] == "--'${arg_name}'")) | .[0][1]'
}

# Parcourir les dossiers commençant par "offline" dans le répertoire défini
for DIR in "$path_offline"offline*; do
    # Vérifier si c'est un dossier
    if [ -d "$DIR" ]; then
        # Chemin vers le fichier wandb-metadata.json
        METADATA_FILE="$DIR/files/wandb-metadata.json"
        
        # Vérifier si le fichier existe
        if [ -f "$METADATA_FILE" ]; then
            # Vérifier si le nom de l'algo est présent dans le fichier
            if grep -q "$ALGO_NAME" "$METADATA_FILE"; then
                # Extraire l'env_id
                ENV_ID=$(extract_arg_value "$METADATA_FILE" "env_id")
                
                echo "Synchronizing $DIR with env_id: $ENV_ID..."
                # wandb sync "$DIR"
                if [ $? -eq 0 ]; then
                    echo "Successfully synchronized $DIR with env_id: $ENV_ID"
                else
                    echo "Failed to synchronize $DIR with env_id: $ENV_ID"
                fi
            fi
        fi
    fi
done

echo "Synchronization process completed."
