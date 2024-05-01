# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/rpo/#rpo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
import ray
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
# import specific 
from src.ce.classifier import Classifier
from envs.wenv import Wenv
from envs.config_env import config
from cmaes import SepCMA
from src.utils.individual import Individual



@dataclass
class Args:
    # XP RECORD
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "contrastive_exploration"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # GIF
    make_gif: bool = True
    """if toggled, will make gif """
    plotly: bool = False
    """if toggled, will use plotly instead of matplotlib"""
    fig_frequency: int = 1

    # ES SPECIFIC
    env_id: str = "HalfCheetah-v3"
    """the id of the environment"""
    total_timesteps: int = 8_000_000
    """total timesteps of the experiment"""
    n_individuals: int = 32
    """the number of individuals"""
    best_individuals: int = 4
    std_optimizer: float = 0.1
    """the standard deviation of the optimizer"""
    nb_epochs: int = 1000
   

    # CLASSIFIER SPECIFIC
    classifier_lr: float = 1e-3
    """the learning rate of the classifier"""
    classifier_epochs: int = 2
    """the number of epochs to train the classifier"""
    classifier_batch_size: int = 256
    """the batch size of the classifier"""
    feature_extractor: bool = False
    """if toggled, a feature extractor will be used"""
    lipshitz: bool = False
    """if toggled, the classifier will be Lipshitz"""
    lipshitz_regu: bool = True
    """if toggled, the classifier will be Lipshitz regularized"""
    epsilon: float = 1e-3
    """the epsilon of the classifier"""
    lambda_init: float = 30.0
    """the lambda of the classifier"""
    bound_spectral: float = 1.0
    """the spectral bound of the classifier"""
    frac_wash: float = 1/2
    """the fraction of the dataset to wash"""
    percentage_time: float = 0/4
    """the percentage of the time to use the classifier"""
    n_iter_lipshitz: int = 1 #1
    """the number of iterations for the Lipshitz constant"""
    clip_lim: float = 100.0
    """the clipping limit of the classifier"""

    # RHO SPECIFIC
    episodic_return: bool = True
    """if toggled, the episodic return will be used"""
    mean_re_init: float = 0.0 #-10.0
    """the mean re-init value"""
    polyak: float = 0.1
    """the polyak averaging coefficient"""
    polyak_speed: float = 0.75
    """ polyak averagieng coefficient for speed """
    n_rollouts: int = 1
    """the number of rollouts"""
    keep_extrinsic_reward: bool = False
    """if toggled, the extrinsic reward will be kept"""
    start_explore: int = 4
    """the number of updates to start exploring"""
    coef_intrinsic : float = 0.1
    """the coefficient of the intrinsic reward"""
    coef_extrinsic : float = 1.0
    """the coefficient of the extrinsic reward"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        env = Wenv(env_id=env_id, xp_id=run_name, **config[env_id])
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env = gym.wrappers.ClipAction(env)
        return env

    return thunk




def update_train(obs_un, obs_un_train, classifier, device, args):
    with torch.no_grad():
        n_batch = int(obs_un_train.shape[0])
        n_replace = int(n_batch*args.frac_wash)
        idx_un = np.random.randint(0, obs_un.shape[0], size = n_batch)
        batch_un = obs_un[idx_un]
        idx_un_train = np.random.randint(0, obs_un_train.shape[0], size = n_batch)
        batch_un_train = obs_un_train[idx_un_train]
        # probs batch un
        batch_probs_un = (torch.sigmoid(classifier(torch.Tensor(batch_un).to(device)))).detach().cpu().numpy().squeeze(-1) 
        batch_probs_un_norm = batch_probs_un/batch_probs_un.sum()
        idx_replace = np.random.choice(idx_un, size=n_replace, p=batch_probs_un_norm)
        # probs un train
        probs_un_train = (1-torch.sigmoid(classifier(torch.Tensor(batch_un_train).to(device)))).detach().cpu().numpy().squeeze(-1) 
        probs_un_train_norm = probs_un_train/probs_un_train.sum()
        idx_remove = np.random.choice(idx_un_train, size=n_replace, p=probs_un_train_norm)
        obs_un_train[idx_remove] = torch.tensor(obs_un[idx_replace], dtype=torch.float).to(device).clone()
    return obs_un_train


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"
    # PLOTTING
    if args.make_gif:
        env_plot = Wenv(env_id=args.env_id, 
                        render_bool_matplot=True, 
                        xp_id=run_name, 
                        **config[args.env_id])
    if args.plotly:
        env_plot = Wenv(env_id=args.env_id, 
                        render_bool_plotly=True, 
                        xp_id=run_name, 
                        **config[args.env_id])
    # coverage check env 
    env_check = Wenv(env_id=args.env_id,
                    render_bool_matplot=False,
                    xp_id=run_name,
                    **config[args.env_id])
    # MAX STEPS
    max_steps = config[args.env_id]['kwargs']['max_episode_steps']
    # BATCH CALCULATION
    # args.classifier_epochs = args.classifier_epochs*args.num_steps // args.classifier_batch_size
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )


    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # CLASSIFIER
    classifier = Classifier(env_check.observation_space, 
                            env_max_steps=max_steps,
                            device=device, 
                            n_agent=1, 
                            lipshitz=args.lipshitz,
                            feature_extractor=args.feature_extractor, 
                            bound_spectral=args.bound_spectral,
                            iter_lip=args.n_iter_lipshitz,
                            lim_up = args.clip_lim,
                            lim_down = -args.clip_lim,
                            env_id=args.env_id, 
                            lipshitz_regu=args.lipshitz_regu,
                            epsilon=args.epsilon,
                            lambda_init=args.lambda_init).to(device)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=args.classifier_lr)

    # DATA STORAGE
    obs_un = None
    dones_un = None
    obs_un_train = torch.tensor(env_check.reset()[0], dtype=torch.float).unsqueeze(0).repeat(max_steps*args.n_individuals, 1).to(device)

    # DKL 
    dkl = args.mean_re_init
   
    # ES
    individual = Individual.remote(env_check)
    optimizer = SepCMA(mean=ray.get(individual.genome.remote()).copy(), sigma=args.std_optimizer, population_size=args.n_individuals+args.best_individuals)
    # best_individuals = [optimizer.ask() for _ in range(args.best_individuals)]
    best_solutions = []
    for update in range(args.total_timesteps):
        #ask genomes
        genomes = [optimizer.ask() for _ in range(args.n_individuals)]
        # evaluate genomes
        results = [individual.eval.remote(genome, env_check) for genome in genomes]
        results = ray.get(results)
        # add buffer 
        rho = np.concatenate([r['data'] for r in results], axis=0)
        rho_dones = np.concatenate([r['dones'] for r in results], axis=0)
        # classifier training 
        if update > 0:
            for _ in range(args.classifier_epochs):
                incremental_epoch = rho.shape[0]//args.classifier_batch_size
                for inc_epoch in range(incremental_epoch):
                    # idx
                    mb_rho_idx = np.random.randint(0, rho.shape[0]-1, args.classifier_batch_size)
                    mb_un_idx =  np.random.randint(0, obs_un.shape[0]-1, args.classifier_batch_size)
                    mb_un_train_idx =  np.random.randint(0, obs_un_train.shape[0]-1, args.classifier_batch_size)
                    # mb
                    mb_obs_rho = torch.Tensor(rho[mb_rho_idx]).to(device)
                    mb_next_obs_rho = torch.Tensor(rho[mb_rho_idx+1]).to(device)
                    mb_rho_dones = torch.Tensor(rho_dones[mb_rho_idx+1]).to(device)
                    mb_obs_un = torch.Tensor(obs_un[mb_un_idx]).to(device)
                    mb_next_obs_un = torch.Tensor(obs_un[mb_un_idx+1]).to(device )
                    mb_un_dones = torch.Tensor(dones_un[mb_un_idx+1]).to(device)
                    mb_obs_un_train = torch.Tensor(obs_un_train[mb_un_train_idx])
                    # loss 
                    loss, _ = classifier.lipshitz_loss_ppo(batch_q= mb_obs_rho, batch_p = mb_obs_un_train,
                                                            q_batch_s = mb_obs_rho, q_batch_next_s = mb_next_obs_rho, q_dones = mb_rho_dones,
                                                            p_batch_s = mb_obs_un, p_batch_next_s = mb_next_obs_un, p_dones = mb_un_dones)
                    classifier_optimizer.zero_grad()
                    loss.backward()
                    classifier_optimizer.step()
                    # lambda loss
                    _, lipshitz_regu = classifier.lipshitz_loss_ppo(batch_q= mb_obs_rho, batch_p = mb_obs_un_train,
                                                            q_batch_s = mb_obs_rho, q_batch_next_s = mb_next_obs_rho, q_dones = mb_rho_dones,
                                                            p_batch_s = mb_obs_un, p_batch_next_s = mb_next_obs_un, p_dones = mb_un_dones)
                    lambda_loss = classifier.lambda_lip*lipshitz_regu
                    classifier_optimizer.zero_grad()
                    lambda_loss.backward()
                    classifier_optimizer.step()
                    # update train
                    obs_un_train = update_train(obs_un, obs_un_train, classifier, device, args).clone()
        # results
        results += best_solutions
        with torch.no_grad():
            for r in results:
                r['fitness'] = -classifier(torch.Tensor(r['data'].copy()).to(device)).mean().item() if not args.keep_extrinsic_reward else -classifier(torch.Tensor(r['data'].copy()).to(device)).mean().item() + args.coef_extrinsic*r['fitness']
        # sort 
        results.sort(key=lambda x: x['fitness'])
        # solutions
        solutions = [(r['genome'], r['fitness']) for r in results]
        # update best solutions
        best_solutions = results[:args.best_individuals].copy()
        # update optimizer
        optimizer.tell(solutions+[(r['genome'], r['fitness']) for r in best_solutions]) if update == 0 else optimizer.tell(solutions)
        # mean 
        mean_epoch = np.mean([-s[1] for s in solutions])
        # std 
        std_epoch = np.std([-s[1] for s in solutions])
        # dkl 
        dkl = args.polyak * mean_epoch + (1-args.polyak) * dkl
        # log
        print('update:', update, 'mean:', mean_epoch, 'std:', std_epoch, 'dkl:', dkl)
        # un update
        if obs_un is None : 
            obs_un = np.concatenate([r['data'][:r['data'].shape[0]//2].copy() for r in results], axis=0)
            dones_un = [r['dones'][:r['data'].shape[0]//2].copy() for r in results]
            for i in range(len(dones_un)): dones_un[i][-1] = 1
            dones_un = np.concatenate(dones_un, axis=0)
        else:
            obs_un = np.concatenate([obs_un.copy(), np.concatenate([r['data'].copy() for r in results[-int(1/2*args.n_individuals):]], axis=0)], axis=0)
            dones_un = np.concatenate([dones_un.copy(), np.concatenate([r['dones'].copy() for r in results[-int(1/2*args.n_individuals):]], axis=0)], axis=0)
    



        

        # # TRY NOT TO MODIFY: record rewards for plotting purposes
        # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # writer.add_scalar("losses/coverage", env_check.get_coverage(), global_step)
        # # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if update % args.fig_frequency == 0 and update > 0:
            if args.make_gif : 
                env_plot.gif(obs_un=obs_un, obs_un_train=obs_un_train, obs_rho=rho, classifier=classifier, device=device)
            if args.plotly:
                env_plot.plotly(obs_un, obs_un_train, classifier, device)

