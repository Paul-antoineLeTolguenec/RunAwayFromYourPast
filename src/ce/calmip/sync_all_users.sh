#!/bin/bash

# Liste des hôtes avec leurs configurations spécifiques
declare -A hosts
hosts=(
    ["hedwin_olympe_sureli"]="127.0.0.1 11300 ~/.ssh/id_rsa bonnavau"
    ["dennis_olympe_sureli"]="127.0.0.1 11300 ~/.ssh/id_rsa p21049wd"
    ["dennis_olympe_dennis"]="127.0.0.1 11300 ~/.ssh/id_rsa p21001wd"
    ["olympe_p21049lp"]="127.0.0.1 11300 ~/.ssh/id_rsa p21049lp"
    ["olympe_letolgue"]="127.0.0.1 11300 ~/.ssh/id_rsa letolgue"
)

# Initialiser les variables pour le résumé global
total_runs=0

# Fonction pour obtenir le nombre de runs sur chaque hôte et afficher la sortie de la commande
get_runs_on_host() {
    local host_name=$1
    local host_info=(${hosts[$host_name]})
    local ip=${host_info[0]}
    local port=${host_info[1]}
    local key=${host_info[2]}
    local user=${host_info[3]}

    # Exécuter la commande sur l'hôte distant en utilisant les options appropriées
    echo "---- Exécution de 'squeue -u $user -o \"%.18i %.9P %.50j %.8u %.2t %.10M %.6D %R\"' sur $host_name ($user@$ip:$port) ----"
    output=$(ssh -i $key -p $port $user@$ip "squeue -u $user")
    echo "$output"

    # Compter le nombre de lignes de "runs" en cours (enlevant l'entête avec `tail -n +2`)
    run_count=$(echo "$output" | tail -n +2 | wc -l)
    echo "Nombre de runs sur $host_name : $run_count"
    
    # Ajouter au total global
    total_runs=$((total_runs + run_count))
    echo
}

# Fonction pour exécuter le script send_wandb.sh sur chaque machine
execute_send_wandb() {
    local host_name=$1
    local host_info=(${hosts[$host_name]})
    local ip=${host_info[0]}
    local port=${host_info[1]}
    local key=${host_info[2]}
    local user=${host_info[3]}

    # Exécuter le script send_wandb.sh sur l'hôte distant
    echo "---- Exécution de 'bash send_wandb.sh' sur $host_name ($user@$ip:$port) ----"
    ssh -i $key -p $port $user@$ip "cd contrastve_exploration/src/ce/calmip/ && bash send_finished_run.sh"
    if [ $? -eq 0 ]; then
        echo "Script send_wandb.sh exécuté avec succès sur $host_name"
    else
        echo "Erreur lors de l'exécution du script send_wandb.sh sur $host_name"
    fi
    echo
}

# Pour chaque hôte, exécute la commande et récupère le nombre de runs
for host in "${!hosts[@]}"; do
    get_runs_on_host $host
    execute_send_wandb $host
done

# Résumé global
echo "Résumé global :"
echo "Nombre total de runs en cours : $total_runs"
