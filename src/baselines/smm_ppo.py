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
import torch.nn.functional as F
# import specific 
from src.ce.classifier import Classifier
from src.ce.vector_encoding import VE
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
    wandb_project_name: str = "contrastive_exploration"
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

    # RPO SPECIFIC
    env_id: str = "Maze-Ur"
    """the id of the environment"""
    total_timesteps: int = 50_000
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
    norm_adv: bool = False
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

    # DIAYN SPECIFIC
    ratio_reward: float = 1.0
    """the ratio of the intrinsic reward"""
    episodic_return: bool = True
    """if toggled, the episodic return will be used"""
    n_rollouts: int = 4
    """the number of rollouts"""
    keep_extrinsic_reward: bool = False
    """if toggled, the extrinsic reward will be kept"""
    coef_intrinsic : float = 1.0
    """the coefficient of the intrinsic reward"""
    coef_extrinsic : float = 1.0
    """ the coefficient of the extrinsic reward"""
    classifier_lr: float = 1e-3
    """the learning rate of the classifier"""
    classifier_epochs: int = 16
    """the number of epochs for the classifier"""
    nb_skill: int = 4
    """the number of agents"""
    classifier_batch_size: int = 256
    """the batch size of the classifier"""
    feature_extractor: bool = True
    """if toggled, the feature extractor will be used"""
    # VAE
    vae_lr: float = 1e-3
    """the learning rate of the VAE"""
    vae_epochs: int = 16
    """the number of epochs for the VAE"""
    vae_latent_dim: int = 32

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # dataset 
    beta_ratio: float = 1/16
    """the ratio of the beta"""
    nb_max_un: int = 256
    """the maximum number of un"""




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

class Classifier(torch.nn.Module):
    def __init__(self, 
                observation_space,
                nb_skill,
                device,
                env_id,
                feature_extractor = False):
        super(Classifier, self).__init__()
        self.relu = torch.nn.ReLU()
        self.nb_skill = nb_skill
        self.env_id = env_id
        self.feature_extractor = feature_extractor  
        if feature_extractor:
            self.fcz1 = torch.nn.Linear(config[env_id]['coverage_idx'].shape[0], 128,device=device)
            self.fcz2 = torch.nn.Linear(128, 64, device=device)
            self.fcz3 =  torch.nn.Linear(64, nb_skill, device=device)
        else:
            self.fcz1 = torch.nn.Linear(observation_space.shape[0], 128,device=device)
            self.fcz2 = torch.nn.Linear(128, 64, device=device)
            self.fcz3 =  torch.nn.Linear(64, nb_skill, device=device)
        self.softmax = torch.nn.Softmax(dim=1)

    
    def forward_z(self, x):
        x = self.feature(x) if self.feature_extractor else x
        x = self.relu(self.fcz1(x))
        x = self.relu(self.fcz2(x))
        return self.fcz3(x)
    
    def mlh_loss(self, obs, z):
        # change dtype to int
        z = z.type(torch.int64)-1
        p_z = self.softmax(self.forward_z(obs))
        p_z_i = torch.sum(p_z*z, dim=-1)
        return -torch.mean(torch.log(p_z_i))
    
    def feature(self, x):
        x=  x[:, :, config[self.env_id]['coverage_idx']] if x.dim() == 3 else x[:, config[self.env_id]['coverage_idx']]
        return x

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(input_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        )
        self.mean_layer = layer_init(nn.Linear(256, latent_dim))
        self.logstd_layer = layer_init(nn.Linear(256, latent_dim))

        self.decoder = nn.Sequential(
            layer_init(nn.Linear(latent_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, input_dim)),
        )

    def encode(self, x):
        x = self.encoder(x)
        mean = self.mean_layer(x)
        logstd = self.logstd_layer(x)
        return mean, logstd

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logstd = self.encode(x)
        z = mean + torch.randn_like(mean) * torch.exp(logstd)
        x_recon = self.decode(z)
        return x_recon, mean, logstd
    
    def loss(self, x, reduce=True):
        x_recon, mean, logstd = self(x)
        recon_loss = F.mse_loss(x_recon, x, reduction='none').sum(-1)
        kl_loss = -0.5 * (1 + 2 * logstd - mean ** 2 - torch.exp(2 * logstd)).sum(-1)
        loss = recon_loss + kl_loss
        if reduce:
            return loss.mean()
        return loss
    
class Agent(nn.Module):
    def __init__(self, envs, nb_skill=1):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod() + nb_skill, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod() + nb_skill , 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x, z):
        return self.critic(torch.cat([x, z], dim=-1))

    def get_action_and_value(self, x, z, action=None):
        action_mean = self.actor_mean(torch.cat([x, z], dim=-1))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(torch.cat([x, z], dim=-1))


def update_un(obs_un, next_obs_un, actions_un, rewards_un,  dones_un, times_un, z_un,
              obs_reshaped, next_obs_reshaped, actions_reshaped, rewards_reshaped, dones_reshaped, times_reshaped, z_reshaped,
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
    z_un[idx_un] = z_reshaped[idx_rho].copy()
    return obs_un, next_obs_un, actions_un, rewards_un, dones_un, times_un, z_un

    
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
    # DIAYN SPECIFIC 
    args.num_envs = args.nb_skill
    z = np.eye(args.nb_skill, dtype=np.float32) 
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
    agent = Agent(envs, args.nb_skill).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    classifier = Classifier(envs.single_observation_space, 
                            args.nb_skill,
                            device,
                            args.env_id,
                            feature_extractor = args.feature_extractor).to(device)
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=args.classifier_lr, eps=1e-5)
    vae = VAE(np.array(envs.single_observation_space.shape).prod(), args.vae_latent_dim).to(device)
    optimizer_vae = optim.Adam(vae.parameters(), lr=args.vae_lr, eps=1e-5)
    replay_buffer = ReplayBuffer(capacity= int(1e6),
                                observation_space= envs.single_observation_space,
                                action_space= envs.single_action_space,
                                device=device)
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs) + (1,)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs) + (1,)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs) + (1,)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)+ (1,)).to(device)
    times = torch.zeros((args.num_steps, args.num_envs)+ (1,)).to(device)
    zs = torch.tensor(z).to(device).unsqueeze(0).repeat(args.num_steps, 1, 1)
    

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
                action, logprob, _, value = agent.get_action_and_value(next_obs, torch.tensor(z).to(device), None)
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
                        wandb.log({"specific/episodic_return": info["episode"]["r"], "specific/episodic_length": info["episode"]["l"], "global_step": global_step})

        ########################### CLASSIFIER TRAINING ###############################
        if obs_un is not None:
            batch_obs = obs.reshape(-1, obs.shape[-1])
            batch_zs = zs.reshape(-1, zs.shape[-1])
            for _ in range(args.classifier_epochs):
                idx = torch.randint(0, batch_obs.shape[0], (args.classifier_batch_size,))
                obs_batch = batch_obs[idx]
                z_batch = batch_zs[idx]
                optimizer_classifier.zero_grad()
                loss = classifier.mlh_loss(obs_batch, z_batch)
                loss.backward()
                optimizer_classifier.step()

        ########################### VAE UPDATE ###############################
            for _ in range(args.vae_epochs):
                idx = torch.randint(0, batch_obs.shape[0], (args.classifier_batch_size,))
                obs_batch = batch_obs[idx]
                vae_loss = vae.loss(obs_batch)
                optimizer_vae.zero_grad()
                vae_loss.backward()
                optimizer_vae.step()

        ########################### REWARD ###############################
        with torch.no_grad():
            # classifier intrinsic reward
            log_p_z = torch.log((torch.softmax(classifier.forward_z(obs), dim=-1)*zs).sum(dim=-1)).unsqueeze(-1) - torch.log(torch.tensor(1/args.nb_skill).float())
            intrinsic_reward_classifier = log_p_z
            # vae 
            intrinsic_reward_vae = vae.loss(obs, reduce = False).unsqueeze(-1)
            # full intrinsic reward 
            intrinsic_reward = intrinsic_reward_classifier + intrinsic_reward_vae

            extrinsic_rewards = rewards
            rewards = intrinsic_reward * args.coef_intrinsic + rewards * args.coef_extrinsic if args.keep_extrinsic_reward else intrinsic_reward*args.coef_intrinsic
        
        ########################### UPDATE UN ###############################
        # permute
        obs_permute = obs.permute(1,0,2)
        times_permute = times.permute(1,0,2)
        actions_permute = actions.permute(1,0,2)
        rewards_permute = extrinsic_rewards.permute(1,0,2)
        dones_permute = dones.permute(1,0,2)
        zs_permute = zs.permute(1,0,2)
        # reshape
        obs_reshaped = obs.reshape(-1, obs_permute.shape[-1]).cpu().numpy()
        actions_reshaped = actions_permute.reshape(-1, actions.shape[-1]).cpu().numpy()
        rewards_reshaped = rewards_permute.reshape(-1).cpu().numpy()
        dones_reshaped = dones_permute.reshape(-1).cpu().numpy()
        zs_reshaped = zs_permute.reshape(-1, zs.shape[-1]).cpu().numpy()
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
                z_un = zs_reshaped[idx_un]

        elif obs_un.shape[0] >= args.num_steps*args.num_envs*args.nb_max_un:
            obs_un, next_obs_un, actions_un, rewards_un, dones_un, times_un, z_un = update_un(obs_un, next_obs_un, actions_un, rewards_un, dones_un, times_un, z_un,
                                                                                            obs_reshaped[:-1], obs_reshaped[1:], actions_reshaped[:-1], rewards_reshaped[:-1], dones_reshaped[:-1], times_reshaped[:-1], zs_reshaped[:-1],
                                                                                            args)
            
        else:
            obs_un = np.concatenate([obs_un, obs_reshaped[idx_un]])
            next_obs_un = np.concatenate([next_obs_un, obs_reshaped[idx_un+1]]) 
            actions_un = np.concatenate([actions_un, actions_reshaped[idx_un]])
            rewards_un = np.concatenate([rewards_un, rewards_reshaped[idx_un]])
            dones_un = np.concatenate([dones_un, dones_reshaped[idx_un]])
            times_un = np.concatenate([times_un, times_reshaped[idx_un]])   
            z_un = np.concatenate([z_un, zs_reshaped[idx_un]])

        

        ########################### PPO UPDATE ###############################
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs, torch.tensor(z).to(device))
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
        b_zs = zs.reshape((-1,) + (args.nb_skill,))
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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_zs[mb_inds] ,b_actions[mb_inds])
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
        })
        # coverage matrix
        normalized_matrix = (env_check.matrix_coverage - env_check.matrix_coverage.min()) / (env_check.matrix_coverage.max() - env_check.matrix_coverage.min()) * 255
        send_matrix(wandb, np.rot90(normalized_matrix), "coverage", global_step)
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