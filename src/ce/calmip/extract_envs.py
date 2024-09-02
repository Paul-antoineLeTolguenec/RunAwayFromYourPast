# Script: extract_envs.py

import sys
import numpy as np

def extract_environments_by_type(config_file, type_id):
    # Charger le fichier de configuration
    config = {}
    exec(open(config_file).read(), config)
    
    # Filtrer les environnements par type_id
    envs_by_type = [env_id for env_id, env in config['config'].items() if env['type_id'] == type_id]
    
    return envs_by_type

if __name__ == "__main__":
    config_file = sys.argv[1]  # Le chemin du fichier de configuration est le premier argument
    type_id = sys.argv[2] if len(sys.argv) > 2 else "maze"  # Le type_id est le deuxième argument, "maze" par défaut
    envs = extract_environments_by_type(config_file, type_id)
    print(" ".join(envs))  # Imprimer les environnements séparés par des espaces pour une utilisation facile dans bash
