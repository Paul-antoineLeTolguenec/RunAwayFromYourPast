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
from src.ce.vector_encoding import VE
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
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
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
    n_rollouts: int = 2
    """the number of rollouts"""
    keep_extrinsic_reward: bool = False
    """if toggled, the extrinsic reward will be kept"""
    coef_intrinsic : float = 1.0
    """the coefficient of the intrinsic reward"""
    coef_extrinsic : float = 1.0
    """ the coefficient of the extrinsic reward"""
    discriminator_lr: float = 1e-3
    """the learning rate of the discriminator"""
    discriminator_epochs: int = 64
    """the number of epochs for the discriminator"""
    n_agent: int = 6
    """the number of agents"""
    discriminator_batch_size: int = 256
    """the batch size of the discriminator"""
    feature_extractor: bool = False
    """if toggled, the feature extractor will be used"""

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

class Discriminator(torch.nn.Module):
    def __init__(self,  
                state_dim, 
                z_dim,
                env_name, 
                featurize,
                device):
        super(Discriminator, self).__init__()
        self.env_name = env_name
        self.l1=torch.nn.Linear(state_dim, 256).to(device)
        self.l2=torch.nn.Linear(256, 64).to(device)
        self.l3=torch.nn.Linear(64, z_dim).to(device)
        # learnable lagrange multiplier
        self.lambda_metra = torch.nn.Parameter(torch.tensor(30.0)) #lambda_metra in the paper
        self.eps = torch.tensor(1e-6).to(device)
    
    def forward(self, s):
        x=torch.nn.functional.relu(self.l1(s))
        x=torch.nn.functional.relu(self.l2(x))
        x=self.l3(x)
        return x

    def lipshitz_loss(self, s, s_next, z):
        phi_s = self(s)
        phi_s_next = self(s_next)
        loss = -(( (phi_s_next - phi_s) * z).sum(dim = -1) + self.lambda_metra * torch.min(self.eps, torch.norm(phi_s_next-phi_s, dim=1) - 1))
        return loss.mean()
    
    def lambda_loss(self, s, s_next, z):
        phi_s = self(s)
        phi_s_next = self(s_next)
        # metrized loss
        loss = -(( (phi_s_next - phi_s) * z).sum(dim = -1) + self.lambda_metra * torch.min(self.eps, torch.norm(phi_s_next-phi_s, dim=1) - 1)).mean()
        return - loss.detach()*self.lambda_metra
        


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
        x_extended = torch.cat([x, z], dim=-1)
        action_mean = self.actor_mean(x_extended)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x_extended)


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
    # NUM ENVS 
    args.num_envs = args.n_agent
    # MAX STEPS
    max_steps = config[args.env_id]['kwargs']['max_episode_steps']
    args.num_steps = max_steps * args.n_rollouts +1
    # BATCH CALCULATION
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    # DEVICE 
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # METRA DISCRETE SKILL : ONE HOT ENCODING
    z = -1/(args.n_agent-1)*torch.ones((args.n_agent, args.n_agent)).to(device) + (1+1/(args.n_agent-1))*torch.eye(args.n_agent).to(device)
    z_one_hot = torch.eye(args.n_agent).to(device)
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

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    # Agent
    agent = Agent(envs, args.n_agent).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    discriminator = Discriminator(state_dim=np.array(envs.single_observation_space.shape).prod(),
                                    z_dim=args.n_agent,
                                    env_name=args.env_id,
                                    featurize=args.feature_extractor,
                                    device=device).to(device)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=args.discriminator_lr, eps=1e-5)
    ve = VE(args.n_agent, device, torch.ones(args.n_agent)/args.n_agent)
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    zs = z.repeat(args.num_steps, 1, 1).to(device)
    z_one_hots = z_one_hot.repeat(args.num_steps, 1, 1).to(device)

    # Full Replay 
    obs_full = None
    z_full = None
    
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    video_filenames = set()
    
    for update in range(1, num_updates + 1):
        for step in range(0, args.num_steps):
            # coverage 
            env_check.update_coverage(next_obs)
            # step 
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # if terminated, reset the env
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, z)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)*args.ratio_reward
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

        ########################### discriminator TRAINING ###############################
        batch_obs = obs.view(-1, *obs.shape[2:])
        batch_zs = zs.view(-1, *zs.shape[2:])
        if obs_full is not None:
            for _ in range(args.discriminator_epochs):
                # lipshitz loss
                idx = torch.randint(0, obs_full.shape[0] - 1, (args.discriminator_batch_size,))
                obs_batch = torch.tensor(obs_full[idx], device=device)
                next_obs_batch = torch.tensor(obs_full[idx+1], device=device)
                z_batch = torch.tensor(z_full[idx], device=device)
                # lipshitz loss
                # idx = torch.randint(0, batch_obs.shape[0] - 1, (args.discriminator_batch_size,))
                # obs_batch = batch_obs[idx]
                # next_obs_batch =   batch_obs[idx+1]
                # z_batch = batch_zs[idx]
                optimizer_discriminator.zero_grad()
                loss = discriminator.lipshitz_loss(obs_batch, next_obs_batch, z_batch)
                loss.backward()
                optimizer_discriminator.step()
                # lambda loss
                optimizer_discriminator.zero_grad()
                lambda_loss = discriminator.lambda_loss(obs_batch, next_obs_batch, z_batch)
                optimizer_discriminator.zero_grad()
                lambda_loss.backward()
                optimizer_discriminator.step()

        ########################### STORE UN ###############################
        obs_full = np.concatenate((obs_full, obs.cpu().numpy().reshape(-1,obs.shape[-1])) , axis = 0) if obs_full is not None else obs.cpu().numpy().reshape(-1,obs.shape[-1])
        z_full = np.concatenate((z_full, zs.cpu().numpy().reshape((-1,) + z.shape[-1:])) , axis = 0) if z_full is not None else zs.cpu().numpy().reshape((-1,) + z.shape[-1:])
        ########################### REWARD ###############################
        with torch.no_grad():
            p_s_z = discriminator.forward(obs)
            intrinsic_reward = (p_s_z*zs).sum(dim=-1)
            # normalized 
            # intrinsic_reward = (intrinsic_reward - intrinsic_reward.mean(dim=1, keepdim=True)) / (intrinsic_reward.std(dim=1, keepdim=True) + 1e-8)
            rewards = intrinsic_reward * args.coef_intrinsic + rewards * args.coef_extrinsic if args.keep_extrinsic_reward else intrinsic_reward*args.coef_intrinsic

        ########################### PPO UPDATE ###############################
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs, z).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                # delta = rewards[t] + args.gamma * nextvalues * nextnonterminal 
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        # normalize advantages
        if args.norm_adv:
            # normalize advantages per skill
            advantages = (advantages - advantages.mean(dim=1, keepdim=True)) / (advantages.std(dim=1, keepdim=True) + 1e-8)
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_zs = zs.reshape((-1,) + z.shape[-1:])
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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_zs[mb_inds] ,b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                # if args.norm_adv:
                #     mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

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
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/advantages_mean", mb_advantages.mean(), global_step)
        # metrics 
        writer.add_scalar("charts/coverage", env_check.get_coverage(), global_step)

        print("SPS:", int(global_step / (time.time() - start_time)))
        print(f"global_step={global_step}")
        print('update : ',update)
        print('coverage : ', env_check.get_coverage())
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.track and args.capture_video:
            for filename in os.listdir(f"videos/{run_name}"):
                if filename not in video_filenames and filename.endswith(".mp4"):
                    wandb.log({f"videos": wandb.Video(f"videos/{run_name}/{filename}")})
                    video_filenames.add(filename)
        if update % args.fig_frequency == 0  and global_step > 0:
            if args.make_gif : 
                env_plot.gif(obs_un = None,
                obs_un_train = None,
                obs = obs,
                classifier = None,
                device = device,
                z_un = None,
                zs=zs,
                obs_rho = None)
            if args.plotly:
                env_plot.plotly(obs_un = obs_full, 
                                obs_un_train = None,
                                classifier = None,
                                device = device,
                                z_un = None)

    envs.close()
    writer.close()
