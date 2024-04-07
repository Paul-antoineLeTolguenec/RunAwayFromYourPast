# contrastive_exploration

# FIRST THING TO DO
* Reward for DUCB : Add Mutual Information reward (?)
* Whopper check 
* setup wenv then get back to baselines



## Exploration experiments
* tabular experiment with gridworld (facultative)
<!-- continuous action -->
* Easy 
* Ur
* Hard 
* Fetch_reach 
* Ant
* Humanoid
* Hopper
<!-- image base -->
* vizdoom / Montezuma's revenge/ pitfall

## Baselines
* Random
* RND sac 
* ICM sac 
* SSM sac 
* APT ? TO DO 
* LSD sac TODO ASAP
* CSD sac TODO ASAP
* METRA sac TODO ASAP
* NGU  sac 
* DIAYN sac 
* RND PPO 
* ICM PPO
* NGU PPO

## Metrics 
* Coverage 
* Complexity


## TIPS XP 
* Maze : 
exp_tau = 0.5
clip_coef = 0.2
clip_coef_mask = 0.4
ent_coef = 0.2
mask_q = None
lipshitz = False
update-epochs = 16
frac = 1/4
NOTE : Equilibrium easy to maintain without mask_q

* MUJOCO :
exp_tau = 0.5
clip_coef = 0.2
clip_coef_mask = 0.4
ent_coef = 0.1
mask_q = 0.5
lipschitz = True
update-epochs = 16
frac = 1/8
NOTE : Equilibrium in favor of q 