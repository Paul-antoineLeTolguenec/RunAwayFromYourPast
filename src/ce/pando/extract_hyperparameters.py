import json
import sys

def extract_hyperparameters(hyperparameters_file, type_id):
    with open(hyperparameters_file, 'r') as file:
        data = json.load(file)
        if type_id in data["hyperparameters"]:
            params = data["hyperparameters"][type_id]
            hyperparams = []
            for key, value in params.items():
                if isinstance(value, bool):
                    value = str(value).lower().capitalize()
                hyperparams.append(f"--{key} {value}")
            print(" ".join(hyperparams))
        else:
            print("")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_hyperparameters.py <hyperparameters_file> <type_id>")
    else:
        extract_hyperparameters(sys.argv[1], sys.argv[2])
