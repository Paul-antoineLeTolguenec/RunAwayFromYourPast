import itertools
import numpy as np
import matplotlib.pyplot as plt

def generate_specific_combinations(dim):
    # Générer les combinaisons où un seul élément est 1, les autres étant 0 ou -1
    combinations = []
    
    # Itérer sur chaque position possible du 1
    for i in range(dim):
        # Générer toutes les combinaisons des éléments restants, qui sont 0 ou -1
        for rest in itertools.product([0, -1], repeat=dim-1):
            # Insérer 1 à la position i
            comb = list(rest[:i]) + [1] + list(rest[i:])
            combinations.append(comb)
    
    return combinations

def plot_vectors(vectors):
    dim = len(vectors[0])
    
    if dim == 2:
        fig, ax = plt.subplots()
        origin = np.zeros(dim)
        
        for vec in vectors:
            ax.quiver(*origin, *vec, angles='xy', scale_units='xy', scale=1)
            ax.text(*vec, f'{vec}', fontsize=12, ha='right')

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        plt.grid()
        plt.show()
    
    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        origin = np.zeros(3)
        
        for vec in vectors:
            ax.quiver(*origin, *vec, color=np.random.rand(3,))
            ax.text(*vec, f'{vec}', fontsize=12, ha='right')
        
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        plt.show()
    
    else:
        print(f"Visualisation possible uniquement pour des dimensions 2D ou 3D, pas pour {dim}D.")

# Paramètre : dimension
dim = 2  # Tu peux changer cette valeur pour une autre dimension

# Générer et afficher toutes les combinaisons
combinations = generate_specific_combinations(dim)

# Affichage des vecteurs
plot_vectors(combinations)
