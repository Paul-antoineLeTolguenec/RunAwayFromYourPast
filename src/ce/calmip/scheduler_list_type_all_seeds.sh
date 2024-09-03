#!/bin/bash

# Fonction pour afficher l'aide
show_help() {
    echo "Usage: $0 --algo <script_algo> --types <type_list> --hp_file <hyper_parameters_file>" 
    echo ""
    echo "Arguments:"
    echo "  --algo    Chemin vers le script d'algorithme (par défaut : ../v1_ppo_kl_adaptive_sampling.py)"
    echo "  --types   Liste des types d'environnements (par défaut : [ \"robotics\", \"mujoco\"])"
    echo "  --hp_file File containing hyperparameters (default: hyper_parameters.json)"
    echo ""
    echo "Exemple: $0 --algo ../v1klsac.py --types \"[robotics, mujoco]\""
}

# Valeurs par défaut
algo="../v1klsac.py"
types=("robotics" "mujoco")
HYPER_PARAMETERS_FILE="hyper_parameters.json"
CONFIG_FILE="../../../envs/config_env.py"

# Analyse des arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --algo)
            algo="$2"
            shift 2
            ;;
        --types)
            # Conversion de la chaîne en liste en retirant les crochets et en séparant par les virgules
            types=($(echo "$2" | tr -d '[]' | tr ',' ' '))
            shift 2
            ;;
        --hp_file)
            HYPER_PARAMETERS_FILE="$2"
            shift 2
            ;;  
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Argument inconnu : $1"
            show_help
            exit 1
            ;;
    esac
done

# Affichage des paramètres obtenus
echo "Algorithme: $algo"
echo "Liste des types d'environnements : ${types[@]}"
echo "Fichier d'hyperparamètres: $HYPER_PARAMETERS_FILE"
echo "Fichier de configuration: $CONFIG_FILE"

# Boucle sur chaque type pour récupérer les environnements associés
for type in "${types[@]}"; do
    # Appel du script Python pour extraire les environnements par type_id
    envs=$(python3 extract_envs.py "$CONFIG_FILE" "$type")
    envs_list=($envs)

    echo "Environnements pour le type \"$type\": ${envs_list[@]}"

    # Boucle sur chaque environnement récupéré
    for env in "${envs_list[@]}"; do
        # Boucle sur chaque seed
        cmd="bash temp_micro_calmip_all_seeds.sh --algo $algo --env_id $env --hp_file $HYPER_PARAMETERS_FILE --wandb_mode offline"
            # execute 
        eval $cmd
    done
done
