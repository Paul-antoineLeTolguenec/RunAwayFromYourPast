#!/bin/bash

# Fonction pour afficher l'aide
show_help() {
    echo "Usage: fichier.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help               Affiche ce message d'aide"
    echo "  --algo ALGO          Chemin vers le fichier de l'algorithme (par défaut: ../v1wsac.py)"
    echo "  --env_id ENV_ID      Identifiant de l'environnement (par défaut: 'HalfCheetah-v3')"
    echo ""
    echo "Description:"
    echo "Ce script exécute une tâche pour un environnement spécifique en utilisant un algorithme spécifié."
}

# Variables par défaut
algo="../v1wsac.py"
env_id="HalfCheetah-v3"

# Traitement des arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --help) show_help; exit 0 ;;
        --algo) algo="$2"; shift ;;
        --env_id) env_id="$2"; shift ;;
        *) echo "Argument inconnu: $1"; show_help; exit 1 ;;
    esac
    shift
done

# Affichage des valeurs pour débogage
echo "algo: $algo"
echo "env_id: $env_id"

# FIND + RUN
algo_id=$(basename "$algo" | sed 's/\.py//')

# Get the path to the config file
CONFIG_FILE="../../../envs/config_env.py"

# Extract type_id for the specified env_id 
type_id=$(awk -v env_id="$env_id" '
    BEGIN { FS="[:,]" }
    $1 ~ env_id { found=1 }
    found && /type_id/ { gsub(/[ "\t]/, "", $2); print $2; exit }
' "$CONFIG_FILE")

if [ "$type_id" != "'atari'" ]; then
    cmd="bash temp_micro_calmip_all_seeds.sh --algo $algo --env_id $env_id --hp_file ../hyper_parameters_sac.json --wandb_mode offline"
    echo "Running: $cmd"
    eval $cmd
else
    echo "Skipping $env_id as it is of type 'atari'"
fi




