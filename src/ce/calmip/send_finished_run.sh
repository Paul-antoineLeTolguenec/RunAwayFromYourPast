#!/bin/bash

# Vérification de l'hostname pour définir le chemin des dossiers offline
echo "Début du script: vérification du hostname..."
HOSTNAME=$(hostname)
echo "Hostname actuel : $HOSTNAME"

if [[ "$HOSTNAME" == *"pando"* ]]; then
    echo "Configuration pour le host 'pando' détectée."
    path_offline="/scratch/disc/p.le-tolguenec/wandb/"
    export WANDB_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_CACHE_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_CONFIG_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_ARTIFACTS_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_RUN_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_DATA_DIR="/scratch/disc/p.le-tolguenec/"
elif [[ "$HOSTNAME" == *"olympe"* ]]; then
    echo "Configuration pour le host 'olympe' détectée."
    if [[ "$USER" != "p21049lp" && "$USER" != "letolgue" ]]; then
        echo "Utilisateur différent de 'p21049lp' et 'letolgue'. Utilisation du chemin /tmpdir/$USER/P_A/"
        path_offline="/tmpdir/$USER/P_A/wandb/"
        export WANDB_DIR="/tmpdir/$USER/P_A/"
        export WANDB_CACHE_DIR="/tmpdir/$USER/P_A/"
        export WANDB_CONFIG_DIR="/tmpdir/$USER/P_A/"
        export WANDB_ARTIFACTS_DIR="/tmpdir/$USER/P_A/"
        export WANDB_RUN_DIR="/tmpdir/$USER/P_A/"
        export WANDB_DATA_DIR="/tmpdir/$USER/P_A/"
    else
        echo "Utilisateur est 'p21049lp' ou 'letolgue'. Utilisation du chemin /tmpdir/$USER/"
        path_offline="/tmpdir/$USER/wandb/"
        export WANDB_DIR="/tmpdir/$USER/"
        export WANDB_CACHE_DIR="/tmpdir/$USER/"
        export WANDB_CONFIG_DIR="/tmpdir/$USER/"
        export WANDB_ARTIFACTS_DIR="/tmpdir/$USER/"
        export WANDB_RUN_DIR="/tmpdir/$USER/"
        export WANDB_DATA_DIR="/tmpdir/$USER/"
    fi
else
    echo "Hostname inconnu : aucune configuration spécifique appliquée."
fi

# Vérifier si au moins un argument est fourni
if [ "$#" -lt 1 ]; then
    echo "Erreur : Aucun nom d'algorithme fourni. Usage: $0 <algorithm_name1> <algorithm_name2> ..."
    exit 1
fi

echo "Algorithmes fournis : ${ALGO_NAMES[@]}"
ALGO_NAMES=("$@")

# Fonction pour extraire la valeur d'un argument spécifique dans wandb-metadata.json
extract_arg_value() {
    local json_file=$1
    local arg_name=$2
    echo "Extraction de la valeur pour l'argument $arg_name dans $json_file"
    grep -A1 "\"--$arg_name\"" "$json_file" | tail -n1 | awk -F'"' '{print $2}'
}

# Fonction pour vérifier si un fichier contient au moins un des algorithmes
contains_algo() {
    local json_file=$1
    echo "Vérification des algorithmes dans $json_file"
    for algo in "${ALGO_NAMES[@]}"; do
        if grep -q "$algo" "$json_file"; then
            echo "Algorithme $algo trouvé dans $json_file"
            return 0
        fi
    done
    echo "Aucun algorithme correspondant trouvé dans $json_file"
    return 1
}

# Fonction pour extraire le statut du run depuis wandb-metadata.json
get_run_status() {
    local json_file=$1
    # echo "Extraction du statut du run depuis $json_file"
    run_status=$(grep -oP '"state":\s*"\K[^"]+' "$json_file")
    echo $run_status
}

count=0
echo "Début de la synchronisation des dossiers dans le chemin : $path_offline"

# Parcourir les dossiers commençant par "offline" dans le répertoire défini
for DIR in "$path_offline"offline*; do
    echo "Vérification du dossier : $DIR"
    if [ -d "$DIR" ]; then
        echo "Dossier $DIR trouvé."
        METADATA_FILE="$DIR/files/wandb-metadata.json"
        echo "Chemin vers wandb-metadata.json : $METADATA_FILE"
        
        if [ -f "$METADATA_FILE" ]; then
            echo "Fichier $METADATA_FILE trouvé."
            if contains_algo "$METADATA_FILE"; then
                RUN_STATUS=$(get_run_status "$METADATA_FILE")
                echo "Statut du run : $RUN_STATUS"
                if [ "$RUN_STATUS" == "finished" ]; then
                    ENV_ID=$(extract_arg_value "$METADATA_FILE" "env_id")
                    echo "Environnement ID extrait : $ENV_ID"
                    count=$((count + 1))
                    echo "Synchronisation de $DIR avec env_id: $ENV_ID..."
                    wandb sync "$DIR"
                    if [ $? -eq 0 ]; then
                        echo "Synchronisation réussie de $DIR avec env_id: $ENV_ID"
                    else
                        echo "Échec de la synchronisation de $DIR avec env_id: $ENV_ID"
                    fi
                else
                    RUN_NAME=$(extract_arg_value "$METADATA_FILE" "run_name")
                    # echo "Le run $RUN_NAME n'est pas terminé (statut: $RUN_STATUS)."
                fi
            fi
        else
            echo "Fichier $METADATA_FILE non trouvé."
        fi
    else
        echo "$DIR n'est pas un dossier valide."
    fi
done

echo "Nombre total de dossiers synchronisés : $count"
