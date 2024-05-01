# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/rpo/#rpo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

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

    # RPO SPECIFIC
    env_id: str = "Hopper-v3"
    """the id of the environment"""
    total_timesteps: int = 8_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    # """the number of parallel game environments"""
    # num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_mask_coef: float = 0.4
    """the mask clipping coefficient"""
    clip_vloss: bool = False #True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    ent_mask_coef: float = 0.05
    """coefficient of the entropy mask"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # CLASSIFIER SPECIFIC
    classifier_lr: float = 1e-3
    """the learning rate of the classifier"""
    classifier_epochs: int =8
    """the number of epochs to train the classifier"""
    classifier_batch_size: int = 256
    """the batch size of the classifier"""
    feature_extractor: bool = False
    """if toggled, a feature extractor will be used"""
    lipshitz: bool = False
    """if toggled, the classifier will be Lipshitz"""
    lipshitz_regu: bool = True
    """if toggled, the classifier will be Lipshitz regularized"""
    epsilon: float = 1e-6
    """the epsilon of the classifier"""
    lambda_init: float = 30.0
    """the lambda of the classifier"""
    bound_spectral: float = 1.0
    """the spectral bound of the classifier"""
    frac_wash: float = 1/1
    """the fraction of the dataset to wash"""
    percentage_time: float = 0/4
    """the percentage of the time to use the classifier"""
    add_percentage_time: float = 4/4
    """the percentage time add rho"""
    n_iter_lipshitz: int = 1 #1
    """the number of iterations for the Lipshitz constant"""
    clip_lim: float = 100.0
    """the clipping limit of the classifier"""

    # RHO SPECIFIC
    episodic_return: bool = True
    """if toggled, the episodic return will be used"""
    n_best_rollout_to_keep: int = 0
    """the number of best rollouts to keep"""
    mean_re_init: float = -1.0 #50.0
    """the mean re-init value"""
    polyak: float = 0.75
    """the polyak averaging coefficient"""
    polyak_speed: float = 0.75
    """ polyak averagieng coefficient for speed """
    n_rollouts: int = 1
    """the number of rollouts"""
    keep_extrinsic_reward: bool = False
    """if toggled, the extrinsic reward will be kept"""
    start_explore: int = 4
    """the number of updates to start exploring"""
    coef_intrinsic : float = 1.0
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, rpo_alpha = 0.0):
        super().__init__()
        self.rpo_alpha = rpo_alpha
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, dkl_rho_un = 0.0, rate_dkl = 0.0):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

def update_train(obs_un, obs_un_train, classifier, device, args):
    with torch.no_grad():
        # n_batch = int(args.classifier_batch_size)
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
    obs_un_train[idx_remove] = obs_un[idx_replace].clone()
    return obs_un_train



def add_to_un(obs_un, 
            dones_un,
              obs,
              obs_rho, 
              actions_rho, 
              logprobs_rho, 
              rewards_rho, 
              dones_rho, 
              values_rho, 
              times_rho, 
              dkl_rho_un,
              rate_dkl,
              classifier,
              args, 
              update,
              nb_rollouts_per_episode):
    # if dkl_rho_un > 0 or True:
    if update >= args.start_explore and dkl_rho_un > 0: 
        # KEEP BEST ROLLOUTS
        list_mean_rollouts = []
        with torch.no_grad():
            for i in range(len(obs_rho)):
                mean_rollout = torch.mean(classifier(obs_rho[i].to(device))).cpu().item()
                list_mean_rollouts.append(mean_rollout)
        ranked_rollouts = np.argsort(list_mean_rollouts)
        args_to_add = []
        # for i in ranked_rollouts[:len(obs_rho)//2]: args_to_add.append(i) #if 0 >= list_mean_rollouts[i]  else None
        for i in ranked_rollouts[:nb_rollouts_per_episode[0]]: args_to_add.append(i) if (0 >= list_mean_rollouts[i]  or obs_un.shape[0] == args.num_steps*args.num_envs) else None
        if len(args_to_add) > 0:
            obs_un = torch.cat([obs_un, torch.cat([obs_rho[i][:int(obs_rho[i].shape[0]*args.add_percentage_time)] for i in args_to_add],dim=0)], dim=0)
            dones_un = torch.cat([dones_un, torch.cat([dones_rho[i][:int(dones_rho[i].shape[0]*args.add_percentage_time)].unsqueeze(-1) for i in args_to_add],dim=0)], dim=0)
            for idx in sorted(args_to_add, reverse=True):
                if 0 >= list_mean_rollouts[idx]:
                    print(f"REMOVE ROLLOUT: {idx}, MEAN: {list_mean_rollouts[idx]}")
                obs_rho.pop(idx)
                actions_rho.pop(idx)
                logprobs_rho.pop(idx)
                rewards_rho.pop(idx)
                dones_rho.pop(idx)
                values_rho.pop(idx)
                times_rho.pop(idx)
            # UPDATE NB ROLLOUTS PER EPISODE
            if len(args_to_add) >= 0 :
                nb_rollouts_per_episode.pop(0) 
            else :
                nb_rollouts_per_episode[0]-= len(args_to_add)
    
        # UPDATE DKL average
        dkl_rho_un = args.mean_re_init
        # dkl_rho_un = 
        # rate_dkl = 0
    return obs_un, dones_un, obs_rho, actions_rho, logprobs_rho, rewards_rho, dones_rho, values_rho, times_rho, dkl_rho_un, rate_dkl, nb_rollouts_per_episode

def select_best_rollout(obs_rho, classifier, args, device):
    list_mean_rollouts = []
    with torch.no_grad():
        for i in range(len(obs_rho)):
            mean_rollout = torch.mean(classifier(obs_rho[i].to(device))).detach().cpu().item()
            list_mean_rollouts.append(mean_rollout)
    ranked_rollouts = np.argsort(list_mean_rollouts)
    best_rollouts = ranked_rollouts[-args.n_best_rollout_to_keep:]
    return best_rollouts

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
    args.num_steps = max_steps * args.n_rollouts +1
    # BATCH CALCULATION
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.classifier_epochs = (args.classifier_epochs*args.num_steps // args.classifier_batch_size) * args.classifier_epochs
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

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    # AGENT
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # CLASSIFIER
    classifier = Classifier(envs.single_observation_space, 
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
    # RPO: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs) + (1,)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs) + (1,)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs) + (1,)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)+ (1,)).to(device)
    times = torch.zeros((args.num_steps, args.num_envs)+ (1,)).to(device)

    # RHO and UN: Storage setup
    # RHO 
    obs_rho = []
    actions_rho = []
    logprobs_rho = []
    rewards_rho = []
    dones_rho = []
    values_rho = []
    times_rho = []
    nb_rollouts_per_episode = []
    # UN
    obs_un_train = torch.tensor(envs.envs[0].reset()[0], dtype=torch.float).unsqueeze(0).repeat(args.num_steps*args.num_envs, 1)
    obs_un = obs_un_train.clone()
    dones_un = torch.zeros((args.num_steps*args.num_envs, 1))

    # INIT DKL_RHO_UN
    dkl_rho_un = args.mean_re_init
    last_dkl_rho_un = args.mean_re_init
    rate_dkl = 0

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    times[0] = torch.tensor(np.array([infos["l"]])).transpose(0,1).to(device)

    for update in range(1, num_updates + 1):
        if args.episodic_return:
            next_obs, infos = envs.reset(seed=args.seed)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.zeros(args.num_envs).to(device)
            num_updates = args.total_timesteps // args.batch_size
            times[0] = torch.tensor(np.array([infos["l"]])).transpose(0,1).to(device)
        
        # PLAYING IN ENV
        for step in range(0, args.num_steps):
            # coverage assessment 
            env_check.update_coverage(next_obs)
            # ppo
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done.unsqueeze(-1)

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, action = None, dkl_rho_un = dkl_rho_un, rate_dkl = rate_dkl)
                # values[step] = value.flatten()
                values[step] = value

            actions[step] = action
            logprobs[step] = logprob.unsqueeze(-1)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            times[step] = torch.tensor(np.array([infos["l"]])).transpose(0,1).to(device)
            done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).unsqueeze(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        # UPDATE RHO 
        obs_flatten = obs.permute(1, 0, 2).reshape(-1, *envs.single_observation_space.shape)
        actions_flatten = actions.permute(1, 0, 2).reshape(-1, *envs.single_action_space.shape)
        logprobs_flatten = logprobs.permute(1, 0, 2).reshape(-1)
        rewards_flatten = rewards.permute(1, 0, 2).reshape(-1)
        dones_flatten = dones.permute(1, 0, 2).reshape(-1)
        values_flatten = values.permute(1, 0, 2).reshape(-1)
        times_flatten = times.permute(1, 0, 2).reshape(-1)
        # ADD TO RHO
        idx_dones = torch.where(dones_flatten)[0].cpu().numpy().tolist()
        idx_dones.append(obs_flatten.shape[0]-1)
        idx_dones.insert(0,0)
        nb_ep = 0
        for i in range(len(idx_dones)-1):
            idx_start = idx_dones[i] + 1
            idx_end = idx_dones[i+1]
            if obs_flatten[idx_start:idx_end].shape[0]!=0:
                nb_ep += 1
                obs_rho.append(obs_flatten[idx_start:idx_end + 1].clone().cpu())
                actions_rho.append(actions_flatten[idx_start:idx_end + 1].clone().cpu())
                logprobs_rho.append(logprobs_flatten[idx_start:idx_end + 1].clone().cpu())
                rewards_rho.append(rewards_flatten[idx_start:idx_end + 1].clone().cpu())
                dones_rho.append(dones_flatten[idx_start:idx_end + 1].clone().cpu())
                values_rho.append(values_flatten[idx_start:idx_end + 1].clone().cpu())
                times_rho.append(times_flatten[idx_start:idx_end + 1].clone().cpu())
        # NB ROLLOUTS PER EPISODE
        print('NB_ROLLOUTS_PER_EPISODE:', nb_ep)
        nb_rollouts_per_episode.append(nb_ep)
        # superior to 0
        if update > args.start_explore:
            # CLASSIFIER TRAINING
            batch_obs_rho = obs.reshape(-1, obs.shape[-1])
            batch_dones_rho = dones.reshape(-1)
            batch_times_rho = times.reshape(-1)
            mask_time = (batch_times_rho > torch.max(batch_times_rho).item()*args.percentage_time).bool()
            batch_obs_rho = batch_obs_rho[mask_time]
            batch_dones_rho = batch_dones_rho[mask_time]
            for epoch in range(args.classifier_epochs):
                mb_rho_idx = np.random.randint(0, batch_obs_rho.shape[0]-1, args.classifier_batch_size)
                mb_rho = batch_obs_rho[mb_rho_idx].to(device)
                mb_un_idx =  np.random.randint(0, obs_un.shape[0]-1, args.classifier_batch_size)
                mb_un_train_idx =  np.random.randint(0, obs_un_train.shape[0]-1, args.classifier_batch_size)
                # mb 
                mb_un_train = obs_un_train[mb_un_train_idx].to(device)
                mb_un = obs_un[mb_un_idx]
                mb_next_rho = batch_obs_rho[mb_rho_idx+1].to(device)
                mb_rho_done = batch_dones_rho[mb_rho_idx].to(device)
                mb_un = mb_un.to(device)
                mb_next_un = obs_un[mb_un_idx+1].to(device) 
                mb_done_un = dones_un[mb_un_idx].to(device)

                # classifier loss + lipshitz regularization
                loss, _ = classifier.lipshitz_loss_ppo(batch_q= mb_rho, batch_p = mb_un_train, 
                                                        q_batch_s = mb_rho, q_batch_next_s = mb_next_rho, q_dones = mb_rho_done,
                                                        p_batch_s = mb_un, p_batch_next_s = mb_next_un, p_dones = mb_done_un)       
                classifier_optimizer.zero_grad()
                loss.backward()
                classifier_optimizer.step()
                # lambda loss
                _, lipshitz_regu = classifier.lipshitz_loss_ppo(batch_q= mb_rho, batch_p = mb_un_train, 
                                                        q_batch_s = mb_rho, q_batch_next_s = mb_next_rho, q_dones = mb_rho_done,
                                                        p_batch_s = mb_un, p_batch_next_s = mb_next_un, p_dones = mb_done_un)     
                lambda_loss = classifier.lambda_lip*lipshitz_regu
                classifier_optimizer.zero_grad()
                lambda_loss.backward()
                classifier_optimizer.step()
                # UPDATE UN TRAIN
                obs_un_train = update_train(obs_un, obs_un_train, classifier, device, args)



        # INTRINSIC REWARD
        with torch.no_grad():
            log_rho_un = classifier(obs)
            # normalize
            # log_rho_un = (log_rho_un - log_rho_un.mean()) / (log_rho_un.std() + 1e-8)
        rewards = args.coef_extrinsic * rewards + args.coef_intrinsic * log_rho_un if args.keep_extrinsic_reward else args.coef_intrinsic * log_rho_un
        mask_pos = (log_rho_un > 0).float()
        # UPDATE DKL average
        # dkl_rho_un = args.polyak * dkl_rho_un + (1-args.polyak) * log_rho_un.mean().item()
        dkl_rho_un = log_rho_un.mean().item()
        # rate_dkl = (dkl_rho_un - last_dkl_rho_un)*(1-args.polyak_speed) + args.polyak_speed*rate_dkl
        last_dkl_rho_un = dkl_rho_un
        # dkl_rho_un = log_rho_un.mean().item()
        rate_dkl = (dkl_rho_un - last_dkl_rho_un)
        print(f"DKL_RHO_UN: {dkl_rho_un}, RATE_DKL: {rate_dkl}")
        print(f'UN SHAPE: {obs_un.shape}, RHO SHAPE: {len(obs_rho)}')
        # args.ent_coef = 0.0 if dkl_rho_un >= 0 else 0.2
        # UPDATE UN
        obs_un, dones_un, obs_rho, actions_rho, logprobs_rho, rewards_rho, dones_rho, values_rho, times_rho,dkl_rho_un, rate_dkl, nb_rollouts_per_episode = add_to_un(obs_un,
                                                                                                                                                dones_un, 
                                                                                                                                                obs,
                                                                                                                                                obs_rho, 
                                                                                                                                                actions_rho, 
                                                                                                                                                logprobs_rho, 
                                                                                                                                                rewards_rho, 
                                                                                                                                                dones_rho, 
                                                                                                                                                values_rho, 
                                                                                                                                                times_rho, 
                                                                                                                                                dkl_rho_un,
                                                                                                                                                rate_dkl,
                                                                                                                                                classifier,
                                                                                                                                                args, 
                                                                                                                                                update, 
                                                                                                                                                nb_rollouts_per_episode)
       
        # UPDATE UN TRAIN
        # obs_un_train = update_train(obs_un, obs_un_train, classifier, device, args)
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(obs.shape[0])):
                if t == obs.shape[0] - 1:
                    nextnonterminal = 1.0 - next_done.unsqueeze(-1)
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        print('mean advantages:', advantages.mean().item())
        print('std advantages:', advantages.std().item())
        print('max advantages:', advantages.max().item())
        print('min advantages:', advantages.min().item())
        print('max rewards:', rewards.max().item())
        print('min rewards:', rewards.min().item())
        print('mean rewards:', rewards.mean().item())
        print('std rewards:', rewards.std().item())
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        if args.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_mask_pos = mask_pos.reshape(-1)
        # Optimizing the policy and value network
        b_inds = np.arange(obs.shape[0])
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, obs.shape[0], args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss3 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_mask_coef, 1 + args.clip_mask_coef)
                pg_loss = (torch.max(pg_loss1, pg_loss2)*(1-b_mask_pos[mb_inds]) + torch.max(pg_loss1, pg_loss3)*b_mask_pos[mb_inds]).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                entropy_mask_loss = (entropy*b_mask_pos[mb_inds]).mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef - args.ent_mask_coef * entropy_mask_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        # explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/coverage", env_check.get_coverage(), global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if update % args.fig_frequency == 0  and global_step > 0:
            if args.make_gif : 
                env_plot.gif(obs_un, obs_un_train, obs, classifier, device, obs_rho = torch.cat(obs_rho, dim=0))
            if args.plotly:
                env_plot.plotly(obs_un, obs_un_train, classifier, device)

    envs.close()
    writer.close()