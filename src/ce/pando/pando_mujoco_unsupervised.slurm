#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --begin=now
#SBATCH --job-name=V1_MUJOCO

# Chargement des modules
module load gcc
module load conda

# Activation de l'environnement conda
conda activate contrastive_env

# Exécution du script Python pour obtenir la liste des environnements mujoco
ENVIRONMENTS=$(python src/utils/extract_envs.py --type_id mujoco)

# Boucle sur chaque environnement récupéré et sur chaque seed
for env_id in $ENVIRONMENTS
do
    for seed in {0..4}
    do
        echo "Lancement de l'environnement ${env_id} avec la seed ${seed}"
        poetry run python ../v1_ppo_lipshitz.py --env_id=$env_id --seed=$seed \
            --total_timesteps=1e8 --track=True \
            --torch_deterministic=True --cuda=True \
            --wandb_project_name="contrastive_exploration" \
            --capture_video=False --make_gif=False --plotly=False \
            --learning_rate=5e-4 --num_envs=1 --anneal_lr=False \
            --gamma=0.99 --gae_lambda=0.95 --num_minibatches=32 \
            --update_epochs=10 --norm_adv=True --clip_coef=0.2 \
            --clip_mask_coef=0.2 --clip_vloss=False --ent_coef=0.01 \
            --ent_mask_coef=0.01 --vf_coef=0.5 --max_grad_norm=0.5 \
            --classifier_lr=1e-3 --classifier_epochs=32 --classifier_batch_size=256 \
            --feature_extractor=False --lipshitz=False --lipshitz_regu=True \
            --epsilon=1e-3 --lambda_init=30.0 --bound_spectral=1.0 \
            --frac_wash=0.25 --percentage_time=0.0 --n_iter_lipshitz=1 \
            --clip_lim=10.0 --episodic_return=True --n_best_rollout_to_keep=0 \
            --mean_re_init=-1.0 --polyak=0.75 --polyak_speed=0.75 \
            --n_rollouts=2 --keep_extrinsic_reward=False --start_explore=4 \
            --coef_intrinsic=1.0 --coef_extrinsic=1.0
    done
done
