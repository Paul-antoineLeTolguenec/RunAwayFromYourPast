import gym
import matplotlib.pyplot as plt

# Créer l'environnement avec le mode de rendu 'rgb_array'
env = gym.make('Hopper-v2', render_mode='rgb_array')
max_steps = env.spec.max_episode_steps
print('Max steps: ', max_steps) 
# Initialiser l'environnement
observation = env.reset()

print('Observation: ', observation)

# # Configurer la fenêtre de matplotlib
# plt.figure()
# img =  env.render()
# window = plt.imshow(img)

# for _ in range(1000):
#     action = env.action_space.sample()  # Choisir une action au hasard
#     env.step(action)  # Appliquer l'action
#     img = env.render()  # Récupérer l'image de l'environnement
#     window.set_data(img)
#     plt.pause(0.001)  # Petite pause pour voir la mise à jour

# plt.close()  
# env.close()
