import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Fonction pour vérifier si une liste est vide
def is_not_empty(obj):
    if isinstance(obj, pd.Series):
        return not obj.empty
    return bool(obj)

# Convertir le dictionnaire en DataFrame
def dict_to_dataframe(experiments_data):
    print('experiments_data:', experiments_data)
    records = []
    for exp_name, envs in experiments_data.items():
        for env_name, seeds in envs.items():
            for seed, metrics in seeds.items():
                coverage = metrics.get('coverage', pd.Series())
                global_step = metrics.get('global_step', pd.Series())
                print('type:', type(coverage))  
                print('type:', type(global_step))
                if is_not_empty(coverage) and is_not_empty(global_step):
                    for c, g in zip(coverage, global_step):
                        records.append({
                            'exp_name': exp_name,
                            'env_name': env_name,
                            'seed': seed,
                            'coverage': c,
                            'global_step': g
                        })
    return pd.DataFrame(records)