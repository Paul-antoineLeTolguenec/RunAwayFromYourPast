import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import gym as gym_old
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
# import specific 
from src.ce.classifier import Classifier
from src.utils.replay_buffer import ReplayBuffer
from src.utils.wandb_utils import send_video, send_matrix, send_dataset

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
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "contrastive_test"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_data: bool = True
    """whether to save the data of the experiment"""

    # GIF
    make_gif: bool = True
    """if toggled, will make gif """
    plotly: bool = False
    """if toggled, will use plotly instead of matplotlib"""
    fig_frequency: int = 1
    """the frequency of plotting figures"""
    shannon_compute_freq: int = 5
    """the frequency of computing shannon entropy"""

    # RPO SPECIFIC
    env_id: str = "Maze-Ur-v0"
    """the id of the environment"""
    total_timesteps: int = 100_000
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
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
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False #True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    rpo_alpha: float = 0.0
    """the alpha parameter for RPO"""

    # NGU SPECIFIC
    ratio_reward: float = 1.0
    """the ratio of the intrinsic reward"""
    episodic_return: bool = True
    """if toggled, the episodic return will be used"""
    n_rollouts: int = 1
    """the number of rollouts"""
    keep_extrinsic_reward: bool = True
    """if toggled, the extrinsic reward will be kept"""
    coef_intrinsic : float = 0.1
    """the coefficient of the intrinsic reward"""
    coef_extrinsic : float = 1.0
    """the coefficient of the extrinsic reward"""
    ngu_lr: float = 1e-3
    """the learning rate of the RND"""
    ngu_epochs: int = 16
    """the number of epochs for the RND"""
    feature_extractor: bool = False
    """if toggled, the feature extractor will be used"""
    clip_intrinsic: float = 10.0
    """the clipping of the intrinsic reward"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # dataset 
    beta_ratio: float = 1/128
    """the ratio of the beta"""
    nb_max_steps: int = 50_000
    """the maximum number of step in un"""




def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        env = Wenv(env_id=env_id, xp_id=run_name, **config[env_id])
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym_old.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym_old.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env = gym.wrappers.ClipAction(env)
        return env

    return thunk

class NGU(nn.Module):   
    def __init__(self, state_dim, action_dim, feature_dim, device, k = 10, c = 0.001, L=5, eps = 1e-3):
        super(NGU, self).__init__()
        # RND
        # trained network
        self.f1 = nn.Linear(state_dim, 128, device=device)
        self.f2 = nn.Linear(128, 64, device=device)
        self.f3 = nn.Linear(64, 1, device=device)
        # target network
        self.f1_t = nn.Linear(state_dim, 128, device=device)
        self.f2_t = nn.Linear(128, 64, device=device)
        self.f3_t = nn.Linear(64, 1, device=device)
        # embedding network
        self.f1_z = nn.Linear(state_dim, 128, device=device)
        self.f2_z = nn.Linear(128, 64, device=device)
        self.f3_z = nn.Linear(64, feature_dim, device=device)
        # action network
        self.f1_a = nn.Linear(feature_dim*2 , 128, device=device)
        self.f2_a = nn.Linear(128, 64, device=device)
        self.f3_a = nn.Linear(64, action_dim, device=device)
        # HP NGU 
        self.k = k
        self.c = c
        self.L = L
        self.epsilon = eps

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = self.f3(x)
        return x
    
    def forward_t(self, x):
        with torch.no_grad():
            x = F.relu(self.f1_t(x))
            x = F.relu(self.f2_t(x))
            x = self.f3_t(x)
            return x
    
    def rnd_loss(self, x, reduce = True):
        return F.mse_loss(self.forward(x), self.forward_t(x)) if reduce else F.mse_loss(self.forward(x), self.forward_t(x), reduction = 'none')
    
    def embedding(self, s):
        x = F.relu(self.f1_z(s))
        x = F.relu(self.f2_z(x))
        x = self.f3_z(x)
        return x
    
    def action_pred(self, s0, s1):
        x = torch.cat([s0, s1], 1)
        x = F.relu(self.f1_a(x))
        x = F.relu(self.f2_a(x))
        x = self.f3_a(x)
        return x
       
    def r_i(self, s, s_episode, s_dm_1, mean_err, std_err):
        rnd_err = self.rnd_loss(s, reduce=False)
        if s_episode.shape[0] > self.k : 
            with torch.no_grad():
                alpha = 1 + (rnd_err - mean_err) / std_err
                s = s.repeat(s_episode.shape[0],1)
                dists = self.distance_matrix_epoch(s, s_episode).unsqueeze(1)
                knn, s_dm = self.sum_k_nearest_epoch(dists, self.k, s_dm_1)
            r_episodic  = 1/(torch.sqrt(knn) + self.c)
            r = r_episodic * torch.min(torch.max(alpha,torch.ones_like(alpha)),torch.ones_like(alpha)*self.L)
            return r.cpu().numpy(), s_dm, rnd_err
        else : 
            return np.array([0.0]), 0.0, rnd_err
    
    def sum_k_nearest_epoch(self, dist, k, s_dm_1):
        k_nearest_neighbors, _ = torch.topk(dist, k=k, dim=0, largest=False)
        k = torch.sum(k_nearest_neighbors, dim = 0)
        k_running = s_dm_1/dist.shape[0]
        s_dm = s_dm_1 + k
        return self.epsilon/(k**2/k_running**2 + self.epsilon), s_dm
    
    def distance_matrix_epoch(self, s, s_epoch):
        x = self.embedding(s)
        x_epoch = self.embedding(s_epoch)
        dist = torch.norm(x - x_epoch, dim=1)
        return dist
    
    def uncertainty_measure(self, s):
        with torch.no_grad():
            alpha = self.rnd_loss(s, reduce=False)
            dist_matrix = self.distance_matrix(s)
            knn = self.sum_k_nearest(dist_matrix, k=self.k)
        r_episodic  = 1/(torch.sqrt(knn) + self.c)
        r = r_episodic * torch.min(torch.max(alpha,torch.ones_like(alpha)),torch.ones_like(alpha)*self.L)
        return r

    def loss(self,s,s_next,a,d): 
        rnd_loss = self.rnd_loss(s)
        # NGU loss 
        s0 = self.embedding(s)
        s1 = self.embedding(s_next)
        h_loss = (self.action_pred(s0, s1) - a)**2 * (1-d)
        return rnd_loss + h_loss.mean()
        
    def distance_matrix(self, s):
        x = self.embedding(s)
        dist = torch.sum(x**2, 1).view(-1, 1) + torch.sum(x**2, 1).view(1, -1) - 2 * torch.mm(x, x.t())
        return torch.sqrt(dist)


    def sum_k_nearest(self, dist_matrix, k=1):
        _, indices = dist_matrix.sort(dim=1)
        k_nearest_indices = indices[:, 1:k+1] 
        k_nearest_values = torch.gather(dist_matrix, 1, k_nearest_indices)
        sum_k_nearest = k_nearest_values.sum(dim=1, keepdim=True)
        mean_k_nearest_2 = sum_k_nearest.mean()**2
        k_2 = sum_k_nearest**2
        sum_k = self.epsilon/(k_2/mean_k_nearest_2 + self.epsilon)
        return sum_k



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
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

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
def update_un(obs_un, next_obs_un, actions_un, rewards_un,  dones_un, times_un,
              obs_reshaped, next_obs_reshaped, actions_reshaped, rewards_reshaped, dones_reshaped, times_reshaped,
              args):
    n_batch = int(obs_un.shape[0]*args.beta_ratio)
    idx_un = np.random.randint(0, obs_un.shape[0], size = n_batch)
    idx_rho = np.random.randint(0, obs_reshaped.shape[0], size = n_batch)
    obs_un[idx_un] = obs_reshaped[idx_rho].copy()
    next_obs_un[idx_un] = next_obs_reshaped[idx_rho].copy()
    actions_un[idx_un] = actions_reshaped[idx_rho].copy()
    rewards_un[idx_un] = rewards_reshaped[idx_rho].copy()
    dones_un[idx_un] = dones_reshaped[idx_rho].copy()
    times_un[idx_un] = times_reshaped[idx_rho].copy()
    return obs_un, next_obs_un, actions_un, rewards_un, dones_un, times_un


if __name__ == "__main__":
    from src.utils.argparse_test import parse_args
    args = parse_args(Args)
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
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
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
    # Agent
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    ngu = NGU(envs.single_observation_space.shape[0], envs.single_action_space.shape[0], 64, device)
    sdm = np.zeros(args.num_envs)
    rnd_err = np.zeros((args.num_steps, args.num_envs) + (1,))
    last_d_idx = np.zeros(args.num_envs, dtype=int)
    optimizer_ngu = optim.Adam(ngu.parameters(), lr=args.ngu_lr, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs) + (1,)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs) + (1,)).to(device)
    extrinsic_rewards = torch.zeros((args.num_steps, args.num_envs) + (1,)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs) + (1,)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)+ (1,)).to(device)
    times = torch.zeros((args.num_steps, args.num_envs)+ (1,)).to(device)

    # UN 
    obs_un = None
    next_obs_un = None
    actions_un = None
    rewards_un = None
    dones_un = None
    times_un = None
    
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    times[0] = torch.tensor(np.array([infos["l"]])).transpose(0,1).to(device)
    intrinsic_reward = np.zeros(args.num_envs)
    for update in range(1, num_updates + 1):
        if args.episodic_return:
            next_obs, infos = envs.reset(seed=args.seed)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.zeros(args.num_envs).to(device)
            num_updates = args.total_timesteps // args.batch_size
            times[0] = torch.tensor(np.array([infos["l"]])).transpose(0,1).to(device)
            # NGU SPECIFIC
            sdm = np.zeros(args.num_envs)
            last_d_idx = np.zeros(args.num_envs, dtype=int)
        
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
                action, logprob, _, value = agent.get_action_and_value(next_obs, action = None)
                # values[step] = value.flatten()
                values[step] = value

            actions[step] = action
            logprobs[step] = logprob.unsqueeze(-1)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            # clip the reward
            reward = np.clip(reward, -1, 1)
            extrinsic_rewards[step] = torch.tensor(reward).to(device).unsqueeze(-1).clone()
            ########################### INTRINSIC REWARD ###############################
            for idx in range(args.num_envs):
                with torch.no_grad():
                    # rewards NGU 
                    obs_episode = obs[last_d_idx[idx]:step+1, idx].cpu().numpy()
                    intrinsic_reward, sdm[idx], rnd_err[step,idx]= ngu.r_i(torch.Tensor(next_obs[idx]).to(device), torch.Tensor(obs_episode).to(device), sdm[idx], np.mean(rnd_err[step,idx]), np.std(rnd_err[step,idx]))
                    clipped_intrinsic_reward = intrinsic_reward
                    reward[idx] = clipped_intrinsic_reward[0] 

            times[step] = torch.tensor(np.array([infos["l"]])).transpose(0,1).to(device)
            done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).unsqueeze(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            if "final_info" in infos:
                for info, idx_env in zip(infos["final_info"], np.where(done)[0]):
                    if info and "episode" in info:
                        # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        wandb.log({"specific/episodic_return": info["episode"]["r"], "specific/episodic_length": info["episode"]["l"], "global_step": global_step})
            # NGU SPECIFIC
            if True in done:
                for idx_env in np.where(done)[0]:
                    sdm[idx_env] = 0.0
                    last_d_idx[idx_env] = step
                    
            

        ########################### NGU UPDATE ###############################
        obs_reshaped = obs.permute(1,0,2).reshape(-1, obs.shape[-1])[:-1]
        actions_reshaped = actions.permute(1,0,2).reshape(-1, actions.shape[-1])[:-1]
        next_obs_reshaped = obs.permute(1,0,2).reshape(-1, obs.shape[-1])[1:]
        dones_reshaped = dones.permute(1,0,2).reshape(-1, dones.shape[-1])[:-1]
        for _ in range(args.ngu_epochs):
            idx_mb = np.random.randint(0, obs_reshaped.shape[0], args.minibatch_size)
            obs_reshaped_mb = obs_reshaped[idx_mb]
            actions_reshaped_mb = actions_reshaped[idx_mb]
            next_obs_reshaped_mb = next_obs_reshaped[idx_mb]
            dones_reshaped_mb = dones_reshaped[idx_mb]
            ngu_loss = ngu.loss(torch.Tensor(obs_reshaped_mb).to(device),
                                torch.Tensor(next_obs_reshaped_mb).to(device),
                                torch.Tensor(actions_reshaped_mb).to(device),
                                torch.Tensor(dones_reshaped_mb).to(device))
            optimizer_ngu.zero_grad()
            ngu_loss.backward()
            optimizer_ngu.step()
        
        ########################### UPDATE UN ###############################
        # permute
        obs_permute = obs.permute(1,0,2)
        times_permute = times.permute(1,0,2)
        actions_permute = actions.permute(1,0,2)
        rewards_permute = extrinsic_rewards.permute(1,0,2)
        dones_permute = dones.permute(1,0,2)
        # reshape
        obs_reshaped = obs.reshape(-1, obs_permute.shape[-1]).cpu().numpy()
        actions_reshaped = actions_permute.reshape(-1, actions.shape[-1]).cpu().numpy()
        rewards_reshaped = rewards_permute.reshape(-1).cpu().numpy()
        dones_reshaped = dones_permute.reshape(-1).cpu().numpy()
        times_reshaped = times_permute.reshape(-1).cpu().numpy()
        # update un
        idx_un = np.random.randint(0, obs_reshaped.shape[0]-1, int(args.beta_ratio*obs_reshaped.shape[0]))
        if obs_un is None:
                obs_un = obs_reshaped[idx_un]
                next_obs_un = obs_reshaped[idx_un+1]
                actions_un = actions_reshaped[idx_un]
                rewards_un = rewards_reshaped[idx_un]
                dones_un = dones_reshaped[idx_un]
                times_un = times_reshaped[idx_un]

        elif obs_un.shape[0] >= args.nb_max_steps:
            obs_un, next_obs_un, actions_un, rewards_un, dones_un, times_un = update_un(obs_un, next_obs_un, actions_un, rewards_un, dones_un, times_un,
                                                    obs_reshaped[:-1], obs_reshaped[1:], actions_reshaped[:-1], rewards_reshaped[:-1], dones_reshaped[:-1], times_reshaped[:-1], 
                                                    args)
            
        else:
            obs_un = np.concatenate([obs_un, obs_reshaped[idx_un]])
            next_obs_un = np.concatenate([next_obs_un, obs_reshaped[idx_un+1]]) 
            actions_un = np.concatenate([actions_un, actions_reshaped[idx_un]])
            rewards_un = np.concatenate([rewards_un, rewards_reshaped[idx_un]])
            dones_un = np.concatenate([dones_un, dones_reshaped[idx_un]])
            times_un = np.concatenate([times_un, times_reshaped[idx_un]])   

        
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1)
        rewards = extrinsic_rewards*args.coef_extrinsic + rewards*args.coef_intrinsic if args.keep_extrinsic_reward else rewards*args.coef_intrinsic
        print('max : ', rewards.max())
        print('min : ', rewards.min())
        ########################### PPO UPDATE ###############################
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

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
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
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

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

                entropy_loss = entropy
                entropy_loss = entropy_loss.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
     
        # compute shannon entropy and coverage on mu 
        if update % args.shannon_compute_freq == 0:
            shannon_entropy_mu, coverage_mu = env_check.get_shanon_entropy_and_coverage_mu(obs_un)
            wandb.log({"specific/shannon_entropy_mu": shannon_entropy_mu, "specific/coverage_mu": coverage_mu, "global_step": global_step})
            
        # metric
        wandb.log({
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
            "charts/advantages_mean": mb_advantages.mean(),
            "specific/coverage": env_check.get_coverage(),
            "specific/shanon_entropy": env_check.shanon_entropy(),
            "charts/SPS": int(global_step / (time.time() - start_time)),
            "global_step": global_step,
            "update": update,
            "specific/rewards_max": rewards.max().item(),
            "specific/rewards_min": rewards.min().item(),
            "specific/rewards_mean": rewards.mean().item(),
            "specific/rewards_std": rewards.std().item(),
            # loss 
            "losses/ngu_loss": ngu_loss.item() if obs_un is not None else 0.0,
        })
        # coverage matrix
        if args.make_gif:
            # coverage matrix
            if env_check.matrix_coverage.ndim > 2:
            # Sum over all dimensions except the first two
                reduced_matrix = env_check.matrix_coverage
                for axis in reversed(range(2, reduced_matrix.ndim)):
                    reduced_matrix = np.sum(reduced_matrix, axis=axis)
            else : 
                reduced_matrix = env_check.matrix_coverage
            normalized_matrix = (reduced_matrix - reduced_matrix.min()) / (reduced_matrix.max() - reduced_matrix.min()) * 255
            send_matrix(wandb, np.rot90(normalized_matrix), "coverage", global_step) if update % args.fig_frequency == 0 else None
        # log 
        print('shanon : ', env_check.shanon_entropy())
        print("SPS:", int(global_step / (time.time() - start_time)))
        print(f"global_step={global_step}")
        print('update : ',update)
        print('coverage : ', env_check.get_coverage())  

        if update % args.fig_frequency == 0  and global_step > 0:
            if args.make_gif : 
                image = env_plot.gif(obs_un = obs_un,
                                obs=obs,
                                    classifier = None,
                                    device= device)
                send_matrix(wandb, image, "gif", global_step)
            if args.plotly:
                env_plot.plotly(obs_un = obs_un, 
                                classifier = None,
                                device = device)
    if args.save_data:
        # dataset
        send_dataset(wandb, obs_un, actions_un, rewards_un, next_obs_un, dones_un, times_un, "dataset", global_step)
    envs.close()