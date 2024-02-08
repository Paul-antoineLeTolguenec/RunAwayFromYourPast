from gym.envs.registration import register

# Enregistrez chaque variante de l'environnement avec un ID unique et les arguments correspondants
register(
    id='Maze-Easy',
    entry_point='contrastive_exploration.envs.continuous_maze:Maze',
    kwargs={'name': 'Easy'} 
)



