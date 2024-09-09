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

# Vérifier si le chemin défini par path_offline existe
if [ -d "$path_offline" ]; then
    echo "Recherche des dossiers contenant 'offline' dans $path_offline..."

    # Utiliser une boucle for pour parcourir les dossiers
    for dir in "$path_offline"*/; do
        # Vérifier si le nom du dossier contient "offline"
        if [[ "$dir" == *offline* ]]; then
            tmp_path="${dir}files/wandb-metadata.json"
            RUN_STATUS=$(get_run_status "$tmp_path")
            echo "Statut du run : $RUN_STATUS"
            # if [ "$RUN_STATUS" == "finished" ]; then
        fi
    done
else
    echo "Le chemin $path_offline n'existe pas."
fi
