# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from envs.wenv import Wenv
from envs.config_env import config
from src.utils.wandb_utils import send_matrix
from src.ce.classifier import Classifier
from scipy.stats import bernoulli


@dataclass
class Args:
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
    wandb_project_name: str = "contrastive_test_2"
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
    fig_frequency: int = 1000
    """the frequency of logging the figures"""
    shannon_compute_freq: int = 5
    """the frequency of computing shannon entropy"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v3"
    """the environment id of the task"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    num_envs: int = 1
    """the number of parallel environments"""
    sac_training_steps: int = 1
    """the number of training steps in each SAC training loop"""
    learning_frequency: int = 1
    """the frequency of training the SAC"""
     
    # Wassesrstein distance specific arguments
    lr_lambda: float = 1e-1
    """the learning rate of the lambda"""
    lr_discriminator: float = 1e-4
    """the learning rate of the discriminator"""
    epsilon: float = 1e-3
    """the epsilon parameter of the wasserstein distance"""
    lambda_init: float = 100.0
    """the initial value of the lambda"""
    lip_cte: float = 1.0 # 0.1 if maze 
    """the constant of the lipschitz"""
    beta_ratio: float = 1/16 #1/16 if maze
    """the ratio of the beta"""
    nb_episodes_rho: int = 4
    """the number of episodes for the rho"""
    pad_rho: int = 4
    """the padding of the rho"""

    # rewards specific arguments
    keep_extrinsic_reward: bool = False
    """if toggled, the extrinsic reward will be kept"""
    coef_extrinsic: float = 1.0
    """the coefficient of the extrinsic reward"""
    coef_intrinsic: float = 1.0
    """the coefficient of the intrinsic reward"""
    


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        env = Wenv(env_id=env_id, xp_id=run_name, **config[env_id])
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class Discriminator(nn.Module):
    def __init__(self, env, lambda_init, epsilon, lip_cte):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        # parameter
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32))
        self.epsilon = torch.tensor(epsilon, dtype=torch.float32)
        self.lip_cte = torch.tensor(lip_cte, dtype=torch.float32)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def constraint(self, mb_obs, mb_next_obs, mb_dones):
        L = (torch.min(self.epsilon,self.lip_cte-torch.norm(self(mb_obs)-self(mb_next_obs), dim=-1))*(1-mb_dones))
        return -L.mean()
    def loss(self, mb_obs_rho, mb_obs_un):
        return self(mb_obs_un).mean() - self(mb_obs_rho).mean()
        # return self.sigmoid(self(mb_obs_un)).mean() - self.sigmoid(self(mb_obs_rho)).mean()
    
        
    
    
if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    discriminator = Discriminator(envs, args.lambda_init, args.epsilon, args.lip_cte).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    discriminator_optimizer = optim.Adam(list(discriminator.parameters()), lr=args.lr_discriminator)
    lambda_optimizer = optim.Adam([discriminator.lambda_param], lr=args.lr_lambda)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    # specific un 
    max_step = config[args.env_id]['kwargs']['max_episode_steps']
    size_un = max_step *  args.nb_episodes_rho
    fixed_idx_un = np.array([], dtype=int)
    # specific rho
    size_rho = max_step * args.nb_episodes_rho


    nb_episodes = 0
    start_time = time.time()
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
                nb_episodes += 1

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        # decide whether to add transition to the un
        if len(fixed_idx_un)<= size_un:
            if bernoulli.rvs(args.beta_ratio):
                fixed_idx_un = np.append(fixed_idx_un, rb.pos-1)
        else : 
            if True in terminations:
                # remove random element
                fixed_idx_un = np.delete(fixed_idx_un, random.randint(0, len(fixed_idx_un)-1))
                # add the last element
                fixed_idx_un = np.append(fixed_idx_un, rb.pos-1)
            else:
                if bernoulli.rvs(args.beta_ratio):
                    # remove random element
                    fixed_idx_un = np.delete(fixed_idx_un, random.randint(0, len(fixed_idx_un)-1))
                    # add the last element
                    fixed_idx_un = np.append(fixed_idx_un, rb.pos-1)
        
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # discriminator training
        if global_step > args.learning_starts and nb_episodes >= (args.nb_episodes_rho + args.pad_rho) :
            # batch un
            batch_inds_un = fixed_idx_un[np.random.randint(0, max(16,len(fixed_idx_un)-args.pad_rho * max_step * args.beta_ratio), args.batch_size)]
            batch_inds_envs_un = np.random.randint(0, args.num_envs, args.batch_size)
            observations_un = torch.Tensor(rb.observations[batch_inds_un, batch_inds_envs_un]).to(device)
            next_observations_un = torch.Tensor(rb.next_observations[batch_inds_un, batch_inds_envs_un]).to(device)
            rewards_un = torch.Tensor(rb.rewards[batch_inds_un, batch_inds_envs_un]).to(device)
            dones_un = torch.Tensor(rb.dones[batch_inds_un, batch_inds_envs_un]).to(device)
            # batch rho 
            batch_inds_rho = np.random.randint(rb.pos-size_rho, rb.pos, args.batch_size)
            batch_inds_envs_rho = np.random.randint(0, args.num_envs, args.batch_size)
            observations_rho = torch.Tensor(rb.observations[batch_inds_rho, batch_inds_envs_rho]).to(device)
            next_observations_rho = torch.Tensor(rb.next_observations[batch_inds_rho, batch_inds_envs_rho]).to(device)
            rewards_rho = torch.Tensor(rb.rewards[batch_inds_rho, batch_inds_envs_rho]).to(device)
            dones_rho = torch.Tensor(rb.dones[batch_inds_rho, batch_inds_envs_rho]).to(device)
            # train the discriminator
            constraints_rho = discriminator.constraint(observations_rho, next_observations_rho, dones_rho)
            constraints_un = discriminator.constraint(observations_un, next_observations_un, dones_un)
            discriminator_loss = discriminator.loss(observations_rho, observations_un) + \
                                discriminator.lambda_param.detach()*(constraints_rho + constraints_un)
            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()
            # train lambda
            lambda_loss = -discriminator.lambda_param*(discriminator.constraint(observations_rho, next_observations_rho, dones_rho) + \
                                discriminator.constraint(observations_un, next_observations_un, dones_un))
            lambda_optimizer.zero_grad()
            lambda_loss.backward()
            lambda_optimizer.step()
            
            if global_step % 100 == 0:
                writer.add_scalar("losses/discriminator_loss", discriminator_loss.item(), global_step)
                writer.add_scalar("losses/lambda_loss", lambda_loss.item(), global_step)
                writer.add_scalar("metrics/constraints_rho", constraints_rho.item(), global_step)
                writer.add_scalar("metrics/constraints_un", constraints_un.item(), global_step)
                writer.add_scalar("metrics/lambda", discriminator.lambda_param.item(), global_step)

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                # rewards
                intrinsic_reward = discriminator(data.next_observations).detach() - discriminator(data.observations).detach()
                # intrinsic_reward = discriminator(data.observations).detach()
                if args.keep_extrinsic_reward:
                    rewards = data.rewards.flatten() * args.coef_extrinsic + intrinsic_reward.flatten() * args.coef_intrinsic
                else:
                    rewards = intrinsic_reward.flatten() * args.coef_intrinsic  
                next_q_value = rewards + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("metrics/rewards_mean", data.rewards.mean().item(), global_step)
                writer.add_scalar("metrics/intrinsic_reward_mean", intrinsic_reward.mean().item(), global_step)
                writer.add_scalar("metrics/intrinsic_reward_max", intrinsic_reward.max().item(), global_step)
                writer.add_scalar("metrics/intrinsic_reward_min", intrinsic_reward.min().item(), global_step)
                print('Elements in un:', len(fixed_idx_un))
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        if global_step % args.fig_frequency == 0  and global_step > 0:
            if args.make_gif : 
                image = env_plot.gif(obs_un = rb.observations[fixed_idx_un],
                                    classifier = discriminator,
                                    device= device)
                send_matrix(wandb, image, "gif", global_step)
            

    envs.close()
    writer.close()