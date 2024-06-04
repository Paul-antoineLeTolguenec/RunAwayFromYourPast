# contrastive_exploration

TODO : 
1/ xp untar (almost done)
2/ update our versions of the algorithme (KL rho and un-k)
3/ why data is not sent properly baselines ? 
3/ beta tune
4/ Offline and online SAC (IQL XP)

# XP 
3 XP : 
* -1 : (datasets, covergage, imgs, complexity)
* -2 : offpolicy initialisation
* -3 : local optimum escape 

# FIRST THING TO DO
* check save dataset and sac working 




<!-- HP -->
v1_kl : 
maze/dmcs/robotics:
beta_min = 1/512
beta_max = 1/64
beta_ratio = 1/128
entropy_mask = 0.05
entropy = 0.05
coef_intrinsic = 1.0
classifier_epoch = 1
norm_adv = True

mujoco:
beta_min = 1/512
beta_max = 1/128
beta_ratio = 1/256
entropy_mask = 0.05
entropy = 0.01
coef_intrinsic = 0.1
classifier_epoch = 1
norm_adv = True

v2_kl :
maze/dmcs/robotics:

## Metrics 
* Coverage 
* Complexity


