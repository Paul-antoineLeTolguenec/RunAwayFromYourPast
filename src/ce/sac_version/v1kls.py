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
import colorednoise as cn


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "contrastive_test_kl"
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
    env_id: str = "Ant-v3"
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
    learning_starts: int = 1e4
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.1
    """Entropy regularization coefficient."""
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""
    num_envs: int = 4
    """the number of parallel environments"""
    learning_frequency: int = 1
    """the frequency of training the SAC"""
     
    # KL specific arguments
    lr_classifier: float = 1e-4
    """the learning rate of the classifier"""
    classifier_batch_size: int = 256
    """the batch size of the classifier"""
    classifier_epochs: int = 1
    """the number of epochs of the classifier"""
    bound_classifier: float = 5.0
    """the bound of the classifier"""
    beta_ratio: float = 1/32 #1/64 if maze
    """the ratio of the beta"""
    nb_episodes_rho: int = 4
    """the number of episodes for the rho"""
    pad_rho: int = 4
    """the padding of the rho"""
    p_custom_noise: float = 0.5
    """the probability of the custom noise"""
    beta_noise: float = 1.0
    """the beta of the noise"""
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
        # self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        # self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        # x = self.ln1(x)
        x = F.relu(self.fc2(x))
        # x = self.ln2(x)
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        # self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        # self.ln2 = nn.LayerNorm(256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.ln1(x)
        x = F.relu(self.fc2(x))
        # x = self.ln2(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, eps = None):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() if eps is None else mean + (std * eps) # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class Classifier(nn.Module):
    def __init__(self, env, bound=5.0):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        self.bound = bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.clamp(self.fc3(x), -self.bound, self.bound)
    
    def loss(self, mb_obs_rho, mb_obs_un ):
        cross_entropy_loss = -(torch.log(self.sigmoid(self(mb_obs_rho))).mean() + torch.log(1-self.sigmoid(self(mb_obs_un))).mean())
        return cross_entropy_loss
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
    classifier = Classifier(envs, bound=args.bound_classifier).to(device)
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    classifier_optimizer = optim.Adam(list(classifier.parameters()), lr=args.lr_classifier)

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
        n_envs=args.num_envs,
    )
    # add time 
    rb.times = np.zeros((args.buffer_size, args.num_envs), dtype=int)
    # specific un 
    max_step = config[args.env_id]['kwargs']['max_episode_steps']
    size_un = max_step *  args.nb_episodes_rho
    fixed_idx_un = np.array([], dtype=int)
    # specific rho
    size_rho = max_step * args.nb_episodes_rho
    nb_rho_episodes = 0
    nb_rho_steps = 0
    start_time = time.time()
    # custom noise 
    eps_tm = np.concatenate([ np.concatenate([cn.powerlaw_psd_gaussian(args.beta_noise, max_step +1 )[:, None] for _ in range(envs.single_action_space.shape[0])], axis=1)[None, :] for _ in range(args.num_envs)], axis=0)
    nb_step_per_env = np.zeros(args.num_envs, dtype=int)    
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        nb_step_per_env += 1
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([np.random.uniform(-max_action, max_action, envs.single_action_space.shape) for _ in range(args.num_envs)])
        else:
            with torch.no_grad():
                eps = eps_tm[np.arange(args.num_envs), nb_step_per_env]
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device), eps=torch.Tensor(eps).to(device))
                actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                try : 
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break
                except:
                    pass

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
            if trunc or terminations[idx]:
                nb_rho_episodes += 1
                # print('eps_tm', eps_tm[idx].shape)
                eps_tm[idx] = np.concatenate([cn.powerlaw_psd_gaussian(args.beta_noise, max_step +1)[:, None] for _ in range(envs.single_action_space.shape[0])], axis=1)[None, :]
                # print('eps_tm', eps_tm[idx].shape)
                # input()
                nb_step_per_env[idx] = 0

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        rb.times[rb.pos-1] = infos['l']
        # decide whether to add transition to the un
        if len(fixed_idx_un)<= size_un:
            if bernoulli.rvs(args.beta_ratio/args.num_envs):
                fixed_idx_un = np.append(fixed_idx_un, rb.pos-1)
        else : 
            if True in terminations:
                # remove random element
                fixed_idx_un = np.delete(fixed_idx_un, random.randint(0, len(fixed_idx_un)-1))
                # add the last element
                fixed_idx_un = np.append(fixed_idx_un, rb.pos-1)
            else:
                if bernoulli.rvs(args.beta_ratio/args.num_envs):
                    # remove random element
                    fixed_idx_un = np.delete(fixed_idx_un, random.randint(0, len(fixed_idx_un)-1))
                    # add the last element
                    fixed_idx_un = np.append(fixed_idx_un, rb.pos-1)
        
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # classifier training
        if global_step*args.num_envs > args.learning_starts and  (global_step*args.num_envs) % size_rho == 0:
            print('global_step', global_step)
            print('nb_rho_episodes', nb_rho_episodes)
            print('nb_rho_steps', nb_rho_steps)
            print('fixed_idx_un', len(fixed_idx_un))
            print('nb discrinimator step', int(size_rho/args.classifier_batch_size * args.classifier_epochs))
            batch_times_rho = rb.times[max(int(rb.pos-size_rho/args.num_envs), 0):rb.pos].transpose(1,0).reshape(-1)
            batch_obs_rho = rb.observations[max(int(rb.pos-size_rho/args.num_envs), 0):rb.pos].transpose(1,0,2).reshape(-1, rb.observations.shape[-1])
            batch_next_obs_rho = rb.next_observations[max(int(rb.pos-size_rho/args.num_envs), 0):rb.pos].transpose(1,0,2).reshape(-1, rb.next_observations.shape[-1])
            batch_dones_rho = rb.dones[max(int(rb.pos-size_rho/args.num_envs), 0):rb.pos].transpose(1,0).reshape(-1)           
            prob = np.clip(1/(1)**(batch_times_rho),0.0, 1_00.0)
            prob = prob/prob.sum()
            for classifier_step in range(int(size_rho/args.classifier_batch_size * args.classifier_epochs)):
                # batch un
                batch_inds_un = fixed_idx_un[np.random.randint(0, max(2,len(fixed_idx_un)-args.pad_rho * max_step * args.beta_ratio), args.classifier_batch_size)]
                batch_inds_envs_un = np.random.randint(0, args.num_envs, args.classifier_batch_size)
                observations_un = torch.Tensor(rb.observations[batch_inds_un, batch_inds_envs_un]).to(device)
                next_observations_un = torch.Tensor(rb.next_observations[batch_inds_un, batch_inds_envs_un]).to(device)
                dones_un = torch.Tensor(rb.dones[batch_inds_un, batch_inds_envs_un]).to(device)
                # batch rho 
                batch_inds_rho = np.random.randint(0, batch_obs_rho.shape[0], args.classifier_batch_size)
                observations_rho = torch.Tensor(batch_obs_rho[batch_inds_rho]).to(device)
                next_observations_rho = torch.Tensor(batch_next_obs_rho[batch_inds_rho]).to(device)
                dones_rho = torch.Tensor(batch_dones_rho[batch_inds_rho]).to(device)
                # train the classifier
                classifier_loss = classifier.loss(observations_rho, observations_un)
                classifier_optimizer.zero_grad()
                classifier_loss.backward()
                classifier_optimizer.step()
               
            
            # if global_step % 100 == 0:
                writer.add_scalar("losses/classifier_loss", classifier_loss.item(), global_step)

        # ALGO LOGIC: training.
        if global_step*args.num_envs > args.learning_starts and global_step % args.learning_frequency == 0:
            # standard sampling
            data = rb.sample(args.batch_size)
            b_observations, b_next_observations, b_actions, b_rewards, b_dones = data.observations, data.next_observations, data.actions, data.rewards, data.dones        
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(b_next_observations)
                qf1_next_target = qf1_target(b_next_observations, next_state_actions)
                qf2_next_target = qf2_target(b_next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                # rewards
                intrinsic_reward = classifier(b_observations).detach()
                intrinsic_reward = torch.clamp(intrinsic_reward, -args.bound_classifier, args.bound_classifier)
                # intrinsic_reward = classifier(b_observations).detach()
                if args.keep_extrinsic_reward:
                    b_rewards = b_rewards.flatten() 
                else:
                    b_rewards = intrinsic_reward.flatten() * args.coef_intrinsic  
                # rewards = b_rewards.flatten() 
                next_q_value = b_rewards + (1 - b_dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(b_observations, b_actions).view(-1)
            # print('qf1_a_values', qf1_a_values.shape)
            qf2_a_values = qf2(b_observations, b_actions).view(-1)
            # print('qf2_a_values', qf2_a_values.shape)
            # print('next_q_value', next_q_value.shape)
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
                    pi, log_pi, _ = actor.get_action(b_observations)
                    qf1_pi = qf1(b_observations, pi)
                    qf2_pi = qf2(b_observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(b_observations)
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
                writer.add_scalar("metrics/rewards_mean", b_rewards.mean().item(), global_step)
                writer.add_scalar("metrics/intrinsic_reward_mean", intrinsic_reward.mean().item(), global_step)
                writer.add_scalar("metrics/intrinsic_reward_max", intrinsic_reward.max().item(), global_step)
                writer.add_scalar("metrics/intrinsic_reward_min", intrinsic_reward.min().item(), global_step)
                # print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        if global_step % args.fig_frequency == 0  and global_step > args.learning_starts:
            if args.make_gif : 
                # print('size rho', size_rho)
                # print('max x rho', rb.observations[max(rb.pos-size_rho, 0):rb.pos][0][:,0].max())
                image = env_plot.gif(obs_un = rb.observations[fixed_idx_un],
                                     obs = rb.observations[max(rb.pos-int(size_rho/args.num_envs), 0):rb.pos], 
                                    classifier = classifier,
                                    device= device)
                send_matrix(wandb, image, "gif", global_step)
            

    envs.close()
    writer.close()