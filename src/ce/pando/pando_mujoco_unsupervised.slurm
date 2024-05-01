#!/bin/bash
#SBATCH --job-name=V1_MUJOCO
#SBATCH --partition=huge
#SBATCH --time=24:00:00
#SBATCH --array=0-39
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=6
#SBATCH --nodes=17  # Minimum de nœuds requis
#SBATCH --ntasks-per-node=3  # Tâches par nœud ajustées pour couvrir toutes les tâches avec 17 nœuds
#SBATCH --mem-per-cpu=4G  # Adaptez cette valeur si nécessaire


# Chargement des modules
module load gcc
module load python/3.9
module load conda

# Activation de l'environnement conda
conda activate bonne_env

# Calcul du index de l'environnement et de la seed à partir de SLURM_ARRAY_TASK_ID
ENV_INDEX=$(($SLURM_ARRAY_TASK_ID / 5))
SEED=$(($SLURM_ARRAY_TASK_ID % 5))

# Exécution du script Python pour obtenir la liste des environnements mujoco
ENVIRONMENTS=$(python src/utils/extract_envs.py --type_id mujoco | tr '\n' ' ')
ARRAY_ENVIRONMENTS=($ENVIRONMENTS)

# Choix de l'environnement basé sur ENV_INDEX
SELECTED_ENV=${ARRAY_ENVIRONMENTS[$ENV_INDEX]}

echo "Lancement de l'environnement ${SELECTED_ENV} avec la seed ${SEED}"
poetry run python ../v1_ppo_lipshitz.py --env_id=$SELECTED_ENV --seed=$SEED \
    --total_timesteps=1e8 --track=True --save=True \
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
