import gym
import matplotlib.pyplot as plt
import inspect

# Créer l'environnement avec le mode de rendu 'rgb_array'
env = gym.make('Hopper-v3', render_mode='rgb_array', reset_noise_scale = 0.0,
               exclude_current_positions_from_observation=True)
max_steps = env.spec.max_episode_steps
print('Max steps: ', max_steps) 
observation,i = env.reset()

print('Observation shape: ', observation.shape)
print('Observation: ', observation)
# # Configurer la fenêtre de matplotlib
# plt.figure()
# img =  env.render()
# window = plt.imshow(img)

for _ in range(1000):
    action = env.action_space.sample()  # Choisir une action au hasard
    env.step(action)  # Appliquer l'action
    # img = env.render()  # Récupérer l'image de l'environnement
    # window.set_data(img)
    # plt.pause(0.0000001)  # Petite pause pour voir la mise à jour

plt.close()  
env.close()
