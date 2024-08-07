import json
import sys

def extract_hyperparameters(hyperparameters_file, type_id, algorithm_id):
    with open(hyperparameters_file, 'r') as file:
        data = json.load(file)
        if type_id in data["hyperparameters"]:
            params = data["hyperparameters"][type_id]
            if algorithm_id in params:
                params = params[algorithm_id]
                hyperparams = []
                for key, value in params.items():
                    if isinstance(value, bool):
                        if value :
                            hyperparams.append(f"--{key}")
                        else : 
                            hyperparams.append(f"--no-{key}")
                    else : 
                        hyperparams.append(f"--{key} {value}")

                print(" ".join(hyperparams))
            else:
                print("")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extract_hyperparameters.py <hyperparameters_file> <type_id> <algorithm_id>")
    else:
        extract_hyperparameters(sys.argv[1], sys.argv[2], sys.argv[3])
