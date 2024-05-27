import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class TabularMaze(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, width, height, start_position, goal_position, walls, transition_matrix=None, render = False  ):
        super(TabularMaze, self).__init__()
        self.width = width
        self.height = height
        self.start_position = start_position
        self.current_position = start_position
        self.goal_position = goal_position
        self.walls = walls
        self.state_visits = np.zeros((width, height), dtype=int)

        self.action_space = spaces.Discrete(4)  # Haut, Bas, Gauche, Droite
        self.observation_space = spaces.Discrete(width * height)

        # Actions: 0 = Haut, 1 = Bas, 2 = Gauche, 3 = Droite
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        # render init
        if render:
            self.figure, self.ax = plt.subplots()
            self._init_render()
        # transfition matrix
            # Transition matrix
        if transition_matrix is None:
            # Si aucune matrice de transition n'est fournie, utiliser des transitions déterministes
            self.transition_matrix = np.eye(4)  # Identité pour des transitions déterministes
        else:
            self.transition_matrix = transition_matrix  # Utiliser la matrice de transition stochastique fournie

    def _init_render(self):
        # Initialisation de la figure et des axes pour le rendu
        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')
        plt.gca().invert_yaxis()  # Inverser l'axe Y pour que l'origine soit en haut à gauche

        # Dessiner la grille
        for x in range(self.width):
            for y in range(self.height):
                self.ax.add_patch(patches.Rectangle((x, y), 1, 1, fill=False, edgecolor='black'))

        # Dessiner les murs
        for wall in self.walls:
            self.ax.add_patch(patches.Rectangle(wall, 1, 1, fill=True, color='black'))

    def step(self, action):
        # Choisir une action en fonction de la matrice de transition
        action_probabilities = self.transition_matrix[action]
        chosen_action = np.random.choice(range(4), p=action_probabilities)
        
        dx, dy = self.actions[action]
        next_position = (self.current_position[0] + dx, self.current_position[1] + dy)

        if 0 <= next_position[0] < self.width and 0 <= next_position[1] < self.height and next_position not in self.walls:
            self.current_position = next_position

        self.state_visits[self.current_position] += 1
        done = self.current_position == self.goal_position
        reward = 1 if done else 0

        return self.current_position, reward, done, {}

    def reset(self):
        self.current_position = self.start_position
        self.state_visits = np.zeros((self.width, self.height), dtype=int)
        self.state_visits[self.current_position] += 1
        return self.current_position

    # def render(self, mode='human'):
    #     maze = np.zeros((self.width, self.height), dtype=str)
    #     maze[:] = ' '
    #     maze[self.walls] = 'X'
    #     maze[self.goal_position] = 'G'
    #     maze[self.current_position] = 'A'
    #     print("\n".join(''.join(row) for row in maze.T))

    def render(self, mode='human'):
        # Supprimer l'agent et l'objectif précédents pour la mise à jour
        [p.remove() for p in reversed(self.ax.patches) if p.get_facecolor() in [(0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 0.0, 1.0)]]

        # Dessiner l'agent
        self.ax.add_patch(patches.Rectangle(self.current_position, 1, 1, fill=True, color='blue'))

        # Dessiner l'objectif
        self.ax.add_patch(patches.Rectangle(self.goal_position, 1, 1, fill=True, color='red'))

        plt.draw()
        plt.pause(0.001)  # Petite pause pour permettre la mise à jour de la figure
        # plt.show()

    def close(self):
        pass

# Initialisation des différentes configurations de labyrinthes
maze_configurations = {
    "maze_1": {
        "width": 5,
        "height": 5,
        "start_position": (0, 0),
        "goal_position": (4, 4),
        "walls": [(1, 1), (2, 2), (3, 3)]
    },
    "maze_2": {
        "width": 6,
        "height": 6,
        "start_position": (0, 5),
        "goal_position": (5, 0),
        "walls": [(1, 4), (2, 3), (3, 2), (4, 1)]
    },
    "maze_3": {
        "width": 7,
        "height": 7,
        "start_position": (3, 6),
        "goal_position": (3, 0),
        "walls": [(1, 5), (2, 4), (4, 2), (5, 1)]
    }
}

# # Création du dictionnaire de labyrinthes
# mazes = {}

# for maze_name, config in maze_configurations.items():
#     mazes[maze_name] = TabularMaze(width=config["width"],
#                                    height=config["height"],
#                                    start_position=config["start_position"],
#                                    goal_position=config["goal_position"],
#                                    walls=config["walls"])


# maze_1
env = TabularMaze(width=maze_configurations["maze_1"]["width"],
                    height=maze_configurations["maze_1"]["height"],
                    start_position=maze_configurations["maze_1"]["start_position"],
                    goal_position=maze_configurations["maze_1"]["goal_position"],
                    walls=maze_configurations["maze_1"]["walls"], 
                    render=True)
s = env.reset()
for _ in range(100):
    a = env.action_space.sample()
    s, r, done, _ = env.step(a)
    env.render()
    if done:
        s = env.reset()