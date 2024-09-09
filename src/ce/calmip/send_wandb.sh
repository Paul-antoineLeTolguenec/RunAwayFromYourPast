#!/bin/bash

# Vérification de l'hostname pour définir le chemin des dossiers offline
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *"pando"* ]]; then
    path_offline="/scratch/disc/p.le-tolguenec/wandb/"
    export WANDB_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_CACHE_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_CONFIG_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_ARTIFACTS_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_RUN_DIR="/scratch/disc/p.le-tolguenec/"
    export WANDB_DATA_DIR="/scratch/disc/p.le-tolguenec/"
elif [[ "$HOSTNAME" == *"olympe"* ]]; then
    if [[ "$USER" != "p21049lp" && "$USER" != "letolgue" ]]; then
        path_offline="/tmpdir/$USER/P_A/wandb/"
        export WANDB_DIR="/tmpdir/$USER/P_A/"
        export WANDB_CACHE_DIR="/tmpdir/$USER/P_A/"
        export WANDB_CONFIG_DIR="/tmpdir/$USER/P_A/"
        export WANDB_ARTIFACTS_DIR="/tmpdir/$USER/P_A/"
        export WANDB_RUN_DIR="/tmpdir/$USER/P_A/"
        export WANDB_DATA_DIR="/tmpdir/$USER/P_A/"
    else
        path_offline="/tmpdir/$USER/wandb/"
        export WANDB_DIR="/tmpdir/$USER/"
        export WANDB_CACHE_DIR="/tmpdir/$USER/"
        export WANDB_CONFIG_DIR="/tmpdir/$USER/"
        export WANDB_ARTIFACTS_DIR="/tmpdir/$USER/"
        export WANDB_RUN_DIR="/tmpdir/$USER/"
        export WANDB_DATA_DIR="/tmpdir/$USER/"
    fi
fi

# get status 
get_run_status() {
    local json_file=$1
    # echo "Extraction du statut du run depuis $json_file"
    run_status=$(grep -oP '"state":\s*"\K[^"]+' "$json_file")
    echo $run_status
}
# get algo 
extract_algo() {
    # Prend le chemin du fichier JSON en argument
    local json_file_path="$1"
    
    # Extrait le champ "program" du fichier JSON
    local program_path=$(grep '"program":' "$json_file_path" | awk -F'"' '{print $4}')
    
    # Vérifie si program_path n'est pas vide
    if [[ -n "$program_path" ]]; then
        # Extrait le nom du fichier sans le chemin et sans l'extension .py
        local algo_name=$(basename "$program_path" .py)
        echo "$algo_name"
    else
        echo "Aucun fichier Python trouvé."
    fi
}
extract_env_id() {
    # Prend le chemin du fichier JSON en argument
    local json_file_path="$1"
    
    # Extrait la valeur de "--env_id" dans le champ "args" du fichier JSON
    local env_id=$(grep -A1 '"--env_id"' "$json_file_path" | tail -n1 | awk -F'"' '{print $2}')
    
    # Vérifie si env_id n'est pas vide
    if [[ -n "$env_id" ]]; then
        echo "$env_id"
    else
        echo "Aucun env_id trouvé."
    fi
}

nb_run=0
# Vérifier si le chemin défini par path_offline existe
if [ -d "$path_offline" ]; then
    echo "Recherche des dossiers contenant 'offline' dans $path_offline..."

    # Utiliser une boucle for pour parcourir les dossiers
    for dir in "$path_offline"*/; do
        # Vérifier si le nom du dossier contient "offline"
        if [[ "$dir" == *offline* ]]; then
            tmp_path="${dir}files/wandb-metadata.json"
            RUN_STATUS=$(get_run_status "$tmp_path")
            ALGO_NAME=$(extract_algo "$tmp_path")
            ENV_ID=$(extract_env_id "$tmp_path")
            echo "Algo name : $ALGO_NAME, env_id : $ENV_ID, run status : $RUN_STATUS"
            if [ "$RUN_STATUS" == "finished" ]; then
                wandb sync $dir
                ((nb_run++))
            fi
        fi
    done
else
    echo "Le chemin $path_offline n'existe pas."
fi

echo "Nombre de script synchronisés : $nb_run"
