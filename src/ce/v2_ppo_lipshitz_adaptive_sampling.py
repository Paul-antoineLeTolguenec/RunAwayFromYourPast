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
    env_id: str = "Maze-Easy"
    """the id of the environment"""
    total_timesteps: int = 8_000_000
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
    clip_mask_coef: float = 0.2
    """the mask clipping coefficient"""
    clip_vloss: bool = False #True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    ent_mask_coef: float = 0.01
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
    percentage_time: float = 0/4
    """the percentage of the time to use the classifier"""
    epsilon: float = 1e-6
    """the epsilon of the classifier"""
    lambda_init: float = 1000.0
    """the lambda of the classifier"""
    bound_spectral: float = 1.0
    """the bound spectral of the classifier"""
    clip_lim: float = 100.0
    """the clipping limit of the classifier"""
    adaptive_sampling: bool = False
    """if toggled, the sampling will be adaptive"""

    

    # RHO SPECIFIC
    episodic_return: bool = True
    """if toggled, the episodic return will be used"""
    polyak: float = 0.75
    """the polyak averaging coefficient"""
    n_rollouts: int = 4
    """the number of rollouts"""
    keep_extrinsic_reward: bool = False
    """if toggled, the extrinsic reward will be kept"""
    start_explore: int = 2
    """the number of updates to start exploring"""
    coef_intrinsic : float = 1.0
    """the coefficient of the intrinsic reward"""
    coef_extrinsic : float = 1.0
    """the coefficient of the extrinsic reward"""
    beta_ratio: float = 1/256
    """the ratio of the beta"""
    nb_max_un: int = 8
    """the number of un"""

    # METRA SPECIFIC
    n_agent: int = 4
    """the number of agents"""
    lambda_im: float = 1.0
    """the lambda of the mutual information"""
    metra_lr: float = 1e-3
    """the learning rate of the metra"""
    metra_epsilon: float = 1e-6
    """the epsilon of the metra"""
    metra_lambda_init: float = 100.0
    """the lambda of the metra"""
    lip_cte: float = 1.0
    """the lip constant"""


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


class Discriminator(torch.nn.Module):
    def __init__(self,  
                state_dim, 
                z_dim,
                env_name, 
                featurize,
                device, 
                lip_cte = 1.0,
                eps = 1e-6,
                lambda_init = 30.0):
        super(Discriminator, self).__init__()
        self.env_name = env_name
        self.l1=torch.nn.Linear(state_dim, 256).to(device)
        self.l2=torch.nn.Linear(256, 64).to(device)
        self.l3=torch.nn.Linear(64, z_dim).to(device)
        # learnable lagrange multiplier
        self.lambda_metra = torch.nn.Parameter(torch.tensor(lambda_init)).to(device) #lambda_metra in the paper
        self.eps = torch.tensor(eps).to(device)
        self.lip_cte = lip_cte
    
    def forward(self, s):
        x=torch.nn.functional.relu(self.l1(s))
        x=torch.nn.functional.relu(self.l2(x))
        x=self.l3(x)
        return x

    def lipshitz_loss(self, s, s_next, z, d):
        phi_s = self(s)
        phi_s_next = self(s_next)
        loss = -(( (phi_s_next - phi_s) * z).sum(dim = -1) + self.lambda_metra.detach() * torch.min(self.eps, self.lip_cte-torch.norm(phi_s-phi_s_next, dim=-1) ))*(1-d)
        return loss.mean()
    
    def lambda_loss(self, s, s_next, z, d):
        phi_s = self(s)
        phi_s_next = self(s_next)
        # metrized loss
        loss = torch.min(self.eps, self.lip_cte-torch.norm(phi_s-phi_s_next, dim=-1))*(1-d)
        return loss.mean().detach()*self.lambda_metra
        

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, n_agent):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod() + n_agent, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod() + n_agent, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x, z):
        x = torch.cat([x, z], dim=-1)
        return self.critic(x)

    def get_action_and_value(self, x, z, action=None):
        action_mean = self.actor_mean(torch.cat([x, z], dim=-1))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(torch.cat([x, z], dim=-1))

def update_probs(obs_un, classifier, device):
    with torch.no_grad():
        # probs batch un
        batch_probs_un = (torch.sigmoid(classifier(torch.Tensor(obs_un).to(device)))).detach().cpu().numpy().squeeze(-1)
        batch_probs_un_norm = batch_probs_un/batch_probs_un.sum()
    return batch_probs_un_norm

def update_un(obs_un, next_obs_un, dones_un, z_un,
              obs_reshaped, next_obs_reshaped, dones_reshaped, zs_reshaped,
              args):
    n_batch = int(obs_un.shape[0]*args.beta_ratio)
    idx_un = np.random.randint(0, obs_un.shape[0], size = n_batch)
    idx_rho = np.random.randint(0, obs_reshaped.shape[0], size = n_batch)
    obs_un[idx_un] = obs_reshaped[idx_rho].copy()
    next_obs_un[idx_un] = next_obs_reshaped[idx_rho].copy()
    dones_un[idx_un] = dones_reshaped[idx_rho].copy()
    z_un[idx_un] = zs_reshaped[idx_rho].copy()
    return obs_un, next_obs_un, dones_un, z_un

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
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
    
    # METRA INIT
    z = -1/(args.n_agent-1)*torch.ones((args.n_agent, args.n_agent)).to(device) + (1+1/(args.n_agent-1))*torch.eye(args.n_agent).to(device)
    z_one_hot = torch.eye(args.n_agent).to(device)
    args.num_envs = args.n_agent

    # MAX STEPS
    max_steps = config[args.env_id]['kwargs']['max_episode_steps']
    args.num_steps = max_steps * args.n_rollouts +1
    # BATCH CALCULATION
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.classifier_epochs = (args.num_steps*args.num_envs // args.classifier_batch_size) * args.classifier_epochs
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


    

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    # AGENT
    agent = Agent(envs, args.n_agent).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # CLASSIFIER
    classifier = Classifier(envs.single_observation_space, 
                            env_max_steps=max_steps,
                            device=device, 
                            n_agent=1, 
                            lipshitz=False,
                            feature_extractor=args.feature_extractor, 
                            lim_up = args.clip_lim,
                            lim_down = -args.clip_lim,
                            env_id=args.env_id, 
                            lipshitz_regu=True,
                            bound_spectral=args.bound_spectral,
                            ).to(device)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=args.classifier_lr)
    # DISCRIMINATOR
    discriminator = Discriminator(state_dim = envs.single_observation_space.shape[0],
                                    z_dim = args.n_agent,
                                    env_name = args.env_id,
                                    featurize = args.feature_extractor,
                                    lip_cte=args.lip_cte,
                                    eps=args.metra_epsilon,
                                    lambda_init=args.metra_lambda_init,
                                    device = device).to(device)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.metra_lr, eps=1e-5)
    # RPO: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs) + (1,)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs) + (1,)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs) + (1,)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)+ (1,)).to(device)
    times = torch.zeros((args.num_steps, args.num_envs)+ (1,)).to(device)
    zs =  torch.tensor(z).to(device).unsqueeze(0).repeat(args.num_steps, 1, 1)
    zs_one_hot =  torch.tensor(z_one_hot).to(device).unsqueeze(0).repeat(args.num_steps, 1, 1)

    # UN
    obs_un = torch.tensor(envs.envs[0].reset()[0], dtype=torch.float).unsqueeze(0).repeat(args.num_steps*args.num_envs, 1).cpu().numpy()
    s1,_,_,_,_ = envs.envs[0].step(envs.envs[0].action_space.sample())
    next_obs_un = torch.tensor(s1, dtype=torch.float).unsqueeze(0).repeat(args.num_steps*args.num_envs, 1).cpu().numpy()
    dones_un = torch.zeros(args.num_steps*args.num_envs).cpu().numpy()
    z_un = torch.tensor(z).repeat(args.num_steps*args.num_envs, 1).cpu().numpy()
    # INIT DKL_RHO_UN
    dkl_rho_un = 0
    last_dkl_rho_un = 0

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
                action, logprob, _, value = agent.get_action_and_value(next_obs, z, action = None)
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
       
        if update > args.start_explore:
            # CLASSIFIER TRAINING + DISCRIMINATOR TRAINING
            batch_obs_rho = obs.reshape(-1, obs.shape[-1])
            batch_dones_rho = dones.reshape(-1)
            batch_times_rho = times.reshape(-1)
            batch_zs_rho = zs.reshape(-1, args.n_agent)
            mask_time = (batch_times_rho > torch.max(batch_times_rho).item()*args.percentage_time).bool()
            batch_obs_rho_masked = batch_obs_rho[mask_time]
            batch_dones_rho_masked = batch_dones_rho[mask_time]
            batch_zs_rho_masked = batch_zs_rho[mask_time]
            for epoch in range(args.classifier_epochs):
                # DISCRIMINATOR
                # mb rho
                mb_rho_idx = np.random.randint(0, batch_obs_rho_masked.shape[0]-1, args.classifier_batch_size)
                mb_obs_rho = batch_obs_rho_masked[mb_rho_idx].to(device)
                mb_next_obs_rho = batch_obs_rho_masked[mb_rho_idx+1].to(device)
                mb_rho_done = batch_dones_rho_masked[mb_rho_idx+1].to(device)
                mb_zs_rho = batch_zs_rho_masked[mb_rho_idx].to(device)
                # mb un 
                if args.adaptive_sampling:
                    probs_un = update_probs(obs_un, classifier, device)
                    idx_un_beta = np.random.choice(np.arange(obs_un.shape[0]), args.classifier_batch_size, p=probs_un)
                else:
                    idx_un_beta = np.random.randint(0, obs_un.shape[0], int(args.classifier_batch_size))
                mb_obs_un = torch.tensor(obs_un[idx_un_beta]).to(device)
                mb_next_obs_un = torch.tensor(next_obs_un[idx_un_beta]).to(device)
                mb_done_un = torch.tensor(dones_un[idx_un_beta]).to(device)
                mb_z_un = torch.tensor(z_un[idx_un_beta]).to(device)
                # classifier loss + lipshitz regularization
                loss, _ = classifier.lipshitz_loss_ppo(batch_q= mb_obs_rho, batch_p = mb_obs_un, 
                                                        q_batch_s =  mb_obs_rho, q_batch_next_s = mb_next_obs_rho, q_dones = mb_rho_done,
                                                        p_batch_s = mb_obs_un, p_batch_next_s = mb_next_obs_un, p_dones = mb_done_un)       
                classifier_optimizer.zero_grad()
                loss.backward()
                classifier_optimizer.step()
                # lambda loss
                _, lipshitz_regu = classifier.lipshitz_loss_ppo(batch_q= mb_obs_rho, batch_p = mb_obs_un, 
                                                        q_batch_s =  mb_obs_rho, q_batch_next_s = mb_next_obs_rho, q_dones = mb_rho_done,
                                                        p_batch_s = mb_obs_un, p_batch_next_s = mb_next_obs_un, p_dones = mb_done_un)       
                classifier_optimizer.zero_grad()
                lambda_loss = classifier.lambda_lip*lipshitz_regu
                classifier_optimizer.zero_grad()
                lambda_loss.backward()
                classifier_optimizer.step()
                # METRA
                
                discriminator_optimizer.zero_grad()
                loss = discriminator.lipshitz_loss(torch.cat([mb_obs_rho, mb_obs_un]), 
                                                   torch.cat([mb_next_obs_rho, mb_next_obs_un]), 
                                                   torch.cat([mb_zs_rho, mb_z_un]),
                                                   torch.cat([mb_rho_done, mb_done_un]))
                
                loss.backward()
                discriminator_optimizer.step()
                # lambda loss
                discriminator_optimizer.zero_grad()
                lambda_loss = discriminator.lambda_loss(torch.cat([mb_obs_rho, mb_obs_un]), 
                                                   torch.cat([mb_next_obs_rho, mb_next_obs_un]), 
                                                   torch.cat([mb_zs_rho, mb_z_un]),
                                                   torch.cat([mb_rho_done, mb_done_un]))
               
                discriminator_optimizer.zero_grad()
                lambda_loss.backward()
                discriminator_optimizer.step()



        # INTRINSIC REWARD
        with torch.no_grad():
            log_rho_un = classifier(obs)
            w_mi = (((discriminator.forward(obs[1:])-discriminator.forward(obs[:-1]))*zs[1:]).sum(dim=-1).unsqueeze(-1))*dones[1:]
            w_mi = torch.cat([w_mi, torch.zeros((1, args.n_agent,1)).to(device)], dim=0) * 10.0
            reward_intrinsic = w_mi
        rewards = args.coef_extrinsic * rewards + args.coef_intrinsic * reward_intrinsic if args.keep_extrinsic_reward else args.coef_intrinsic * reward_intrinsic
        mask_pos = (log_rho_un > 0).float()
        # UPDATE DKL average
        dkl_rho_un = log_rho_un.mean().item()
        # dkl_rho_un = log_rho_un.mean().item()
        rate_dkl = (dkl_rho_un - last_dkl_rho_un)
        print(f"DKL_RHO_UN: {dkl_rho_un}, RATE_DKL: {rate_dkl}")
        
        # UPDATE UN
        print('eval log_rho_un:', ((log_rho_un>=0)*log_rho_un).mean().item())
        # if ((log_rho_un>=0)*log_rho_un).mean() > args.clip_lim * 0.01:
        if log_rho_un.mean().item() > 0 :
            obs_reshaped = obs.reshape(-1, obs.shape[-1]).cpu().numpy()
            dones_reshaped = dones.reshape(-1).cpu().numpy()
            zs_reshaped = zs.reshape(-1, args.n_agent).cpu().numpy()

            if obs_un.shape[0] <=args.nb_max_un*args.num_envs*args.num_steps:
                obs_un = np.concatenate([obs_un, obs_reshaped[:-1]])
                next_obs_un = np.concatenate([next_obs_un, obs_reshaped[1:]])
                dones_un = np.concatenate([dones_un, dones_reshaped[1:]])
                z_un = np.concatenate([z_un, zs_reshaped[1:]])
            else : 
                obs_un, next_obs_un, dones_un, z_un = update_un(obs_un, next_obs_un, dones_un, z_un,
                                                        obs_reshaped[:-1], obs_reshaped[1:], dones_reshaped[1:], zs_reshaped,
                                                        args)
                
        last_dkl_rho_un = dkl_rho_un
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs, z)
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
        b_zs = zs.reshape((-1,) +(args.n_agent,))
    
        # Optimizing the policy and value network
        b_inds = np.arange(obs.shape[0])
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, obs.shape[0], args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_zs[mb_inds],b_actions[mb_inds])
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
                env_plot.gif(obs_un=obs_un, obs=obs, classifier=classifier, device=device)
            if args.plotly:
                env_plot.plotly(obs_un,  classifier, device)

    envs.close()
    writer.close()