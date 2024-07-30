# contrastive_exploration

TODO : 
1/ maze 200k steps 
2/ Adapt beta to max steps 
3/ relaunch all xps 

# XP 
3 XP :
mazes : Easy / Ur / Hard 
mujoco : HalfCheetah / Walker2D / Ant
gym-robotics : FetchReach / FetchSlide / Finger

TODO : 
*CrossQ
*PinkNoise



* -1 : (datasets, covergage, imgs, complexity)
* -2 : offpolicy initialisation
* -3 : local optimum escape 






<!-- HP EXPLORATION ONLY -->
## v1_kl : 
maze/dmcs/robotics:
beta_min = 0.001953125
beta_max = 0.015625
beta_ratio = 0.0078125
adaptive_beta = true
beta_increment_bool = false
entropy_mask = 0.05
entropy = 0.05
coef_intrinsic = 1.0
classifier_epoch = 1
norm_adv = true

mujoco:
beta_min = 0.001953125
beta_max = 0.0078125
beta_ratio = 0.00390625
adaptive_beta = true
beta_increment_bool = false
entropy_mask = 0.05
entropy = 0.05
coef_intrinsic = 0.1
classifier_epoch = 1
norm_adv = false

## v1_lipshitz :
maze/dmcs/robotics:
beta_ratio = 0.0078125
adaptive_beta = false
beta_increment_bool = false
entropy_mask = 0.05
entropy = 0.05
coef_intrinsic = 1.0
classifier_epoch = 1
classifier_batch_size = 256
classifier_lr = 1e-3
lambda_init = 10.0
norm_adv = true

mujoco:
maze/dmcs/robotics:
beta_ratio = 0.00390625
adaptive_beta = false
beta_increment_bool = false
entropy_mask = 0.05
entropy = 0.05
coef_intrinsic = 1.0
classifier_epoch = 1
classifier_batch_size = 128
classifier_lr = 1e-3
lambda_init = 10.0
norm_adv = false

## v2_kl :
maze/dmcs/robotics:
beta_min = 0.001953125
beta_max = 0.0078125
beta_ratio = 0.015625
adaptive_beta = true
beta_increment_bool = false
entropy_mask = 0.05
entropy = 0.05
coef_intrinsic = 1.0
classifier_epoch = 1
classifier_lr = 1e-3
norm_adv = false
start_explore =  4
lambda_im = 1.0
lambda_ent = 1.0

mujoco:
beta_min = 0.001953125
beta_max = 0.0078125
beta_ratio = 0.00390625
adaptive_beta = false
beta_increment_bool = true
entropy_mask = 0.05
entropy = 0.05
coef_intrinsic = 1.0
classifier_epoch = 1
classifier_lr = 1e-4
norm_adv = false
start_explore =  16
lambda_im = 0.5
lambda_ent = 1.0

## v2_lipshitz :

## Metrics 
* Coverage 
* Shanon score
* Complexity
* extrinsic reward maximization with intrinsic reward
* offline


V2 PPO LIPSHITZ WORK