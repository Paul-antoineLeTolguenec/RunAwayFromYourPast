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
from torch.nn.utils import spectral_norm
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from envs.wenv import Wenv
from envs.config_env import config
from src.utils.wandb_utils import send_matrix
from src.ce.classifier import Classifier
from scipy.stats import bernoulli
from src.utils.image_utils import resize_image
import colorednoise as cn


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "run_away_test"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    use_hp_file : bool = False
    """if toggled, will load the hyperparameters from file"""
    hp_file: str = "hyper_parameters_sac.json"
    """the path to the hyperparameters json file"""
    sweep_mode: bool = False
    """if toggled, will log the sweep id to wandb"""

    # GIF
    make_gif: bool = True
    """if toggled, will make gif """
    plotly: bool = False
    """if toggled, will use plotly instead of matplotlib"""
    fig_frequency: int = 1000
    """the frequency of logging the figures"""
    metric_freq: int = 1000
    """the frequency of ploting metric"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v3"
    """the environment id of the task"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e7)
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
    policy_frequency: int = 4
    """the frequency of training policy (delayed)"""
    learning_frequency: int = 2
    """the frequency of training the Q network"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.05
    """Entropy regularization coefficient."""
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""
    num_envs: int = 4
    """the number of parallel environments"""
    
     
    # LSD specific
    beta_noise: float = 0.0
    """the beta of the noise"""
   
    # LSD specific
    nb_skill: int = 4
    """the number of skills"""
    lr_discriminator_LSD: float = 1e-4
    """the learning rate of the discriminator"""
    lambda_reward_LSD: float = 1.0
    """weight for the reward maximizing the wasserstein equivalent of the Mutual Information"""
    epsilon_LSD: float = 1e-3
    """relaxing constant"""
    lr_lambda_LSD: float = 1e-1
    """ lambda LSD learning rate """
    lambda_LSD_init: float = 30.0 
    """ Lagrange parameter initialization """
    LSD_batch_size: int = 256
    """ bath size for LSD  """
    LSD_discriminator_epochs: int = 100
    """ number of epochs for LSD discriminator """
    LSD_max_step: int = 200
    """ max step for LSD """
    episode_per_epoch: int = 8
    """ number of episode per epoch """
    lip_cte_LSD: float = 1.0
    """ the constant of the lipschitz for LSD """
    tau_update: float = 0.001
    """ tau update for mean and std """
    epsilon_LSD: float = 1e-3 
    """ epsilon for LSD """
    nb_rho_episodes: int = 4
    """ number of episodes for rho """
   

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
    def __init__(self, env, nb_skill):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape) + nb_skill, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, z, a):
        x = torch.cat([x, z, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, nb_skill):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + nb_skill, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x, z):
        x = torch.cat([x, z], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std

    def get_action(self, x, z, eps = None):
        mean, log_std = self(x,z)
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

    
class Discriminator_LSD(torch.nn.Module):
    def __init__(self,  
                state_dim, 
                z_dim,
                env_name, 
                device, 
                lip_cte = 1.0):
        super(Discriminator_LSD, self).__init__()
        self.env_name = env_name
        self.l1=spectral_norm(torch.nn.Linear(state_dim, 1024).to(device))
        self.l2=spectral_norm(torch.nn.Linear(1024, 1024).to(device))
        self.l3=spectral_norm(torch.nn.Linear(1024, z_dim).to(device))
        self.lip_cte = lip_cte
    
    def forward(self, s):
        x=torch.nn.functional.relu(self.l1(s))
        x=torch.nn.functional.relu(self.l2(x))
        x=self.l3(x)
        return x

    def lipshitz_loss(self, s, s_next, z, d):
        phi_s = self(s)
        phi_s_next = self(s_next)
        inner_product = ((phi_s_next - phi_s) * z).sum(dim = -1)
        loss = -(inner_product)*(1-d)
        return loss.mean(), inner_product.mean()
    
 

    
    
if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    if args.use_hp_file:
        import json
        with open(args.hp_file, "r") as f:
            type_id = config[args.env_id]['type_id']
            hp = json.load(f)['hyperparameters'][type_id][args.exp_name]
            for k, v in hp.items():
                setattr(args, k, v)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # LSD Specific
    args.num_envs = args.nb_skill
    z = -1/(args.nb_skill-1)*torch.ones((args.nb_skill, args.nb_skill)).to(device) + (1+1/(args.nb_skill-1))*torch.eye(args.nb_skill).to(device)
    z = z/z.norm(dim=0)
    # z = torch.tensor(np.array([[0,1], 
    #                             [0,-1], 
    #                             [1, 0], 
    #                             [-1,0]], dtype=np.float32))
    z_one_hot = torch.eye(args.nb_skill).to(device)


    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        if args.sweep_mode:
            wandb.init()
            # set config from sweep
            wandb.config.update(args)
        else :
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
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

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs, nb_skill= args.nb_skill).to(device)
    qf1 = SoftQNetwork(envs, nb_skill= args.nb_skill).to(device)
    qf2 = SoftQNetwork(envs, nb_skill= args.nb_skill).to(device)
    qf1_target = SoftQNetwork(envs, nb_skill= args.nb_skill).to(device)
    qf2_target = SoftQNetwork(envs, nb_skill= args.nb_skill).to(device)
    discriminator_LSD = Discriminator_LSD(state_dim = np.array(envs.single_observation_space.shape).prod(),
                                                z_dim = args.nb_skill,
                                                env_name = args.env_id,
                                                device = device,
                                                lip_cte = args.lip_cte_LSD).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    discriminator_LSD_optimizer = optim.Adam(list(discriminator_LSD.parameters()), lr=args.lr_discriminator_LSD)

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
    rb.zs = np.zeros((args.buffer_size, args.num_envs, args.nb_skill), dtype=np.float32)
    # specific un 
    max_step = config[args.env_id]['kwargs']['max_episode_steps']
    # specific rho
    size_rho = max_step * args.nb_rho_episodes
    # running episodic return
    running_episodic_return = 0 
    LSD_max = args.LSD_max_step
    # obs : mean + std 
    obs_mean = np.zeros(np.array(envs.single_observation_space.shape).prod(), dtype=np.float32)
    obs_std = np.ones(np.array(envs.single_observation_space.shape).prod(), dtype=np.float32)
    # TRY NOT TO MODIFY: start the game
    start_time = time.time()    
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # coverage assessment 
        env_check.update_coverage(obs)
        # ALGO LOGIC: put action logic here
        if global_step*args.num_envs < args.learning_starts:
            actions = np.array([np.random.uniform(-max_action, max_action, envs.single_action_space.shape) for _ in range(args.num_envs)])
        else:
            with torch.no_grad():
                normalized_obs = (obs - obs_mean) / obs_std
                actions, _, _ = actor.get_action(torch.Tensor(normalized_obs).to(device), torch.tensor(z).to(device), eps=None)
                actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}")
                    wandb.log({
                    "charts/episodic_return" : info["episode"]["r"],
                    "charts/episodic_length" : info["episode"]["l"],
                    }, step = global_step) if args.track else None
                    

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        if (global_step)%1_000_000==0 :
            envs.call("set_max_steps", LSD_max)
            LSD_max  = min(LSD_max + 200, config[args.env_id]['kwargs']['max_episode_steps'])

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        rb.times[rb.pos-1 if not rb.full else rb.buffer_size-1] = infos['l']
        rb.zs[rb.pos-1 if not rb.full else rb.buffer_size - 1] = z.detach().cpu().numpy()
        
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs


        if global_step*args.num_envs > args.learning_starts and  global_step % int((args.LSD_max_step * args.episode_per_epoch)/args.num_envs) == 0:
            for _ in range(args.LSD_discriminator_epochs):
                # LSD training
                batch_inds = np.random.randint(0 ,rb.pos if not rb.full else rb.buffer_size, int(args.LSD_batch_size))                    
                batch_inds_env = np.random.randint(0, args.num_envs, args.LSD_batch_size)
                batch_obs = torch.tensor((rb.observations[batch_inds, batch_inds_env] - obs_mean) / obs_std , device=device)
                batch_next_obs = torch.tensor((rb.next_observations[batch_inds, batch_inds_env] -obs_mean) / obs_std , device=device)
                batch_z = torch.tensor(rb.zs[batch_inds, batch_inds_env], device=device)
                batch_dones = torch.tensor(rb.dones[batch_inds, batch_inds_env], device=device)
                loss_LSD, inner_product_loss = discriminator_LSD.lipshitz_loss(batch_obs, batch_next_obs, batch_z, batch_dones) 
                discriminator_LSD_optimizer.zero_grad()
                loss_LSD.backward()
                discriminator_LSD_optimizer.step()
                wandb.log({
                    # losss
                    "losses_LSD/discriminator_loss": loss_LSD.item(), 
                    "losses_LSD/inner_product_loss": inner_product_loss.item(),
                    }, step = global_step) if args.track else None
                    


        # ALGO LOGIC: training.
        if global_step*args.num_envs > args.learning_starts and global_step % args.learning_frequency == 0:
            # standard sampling
            b_inds = np.random.randint(0, rb.pos if not rb.full else rb.buffer_size, args.batch_size)
            b_inds_envs = np.random.randint(0, args.num_envs, args.batch_size)
            # batch obs + next_obs
            b_observations = rb.observations[b_inds, b_inds_envs] 
            # update mean + std obs
            obs_mean = obs_mean * (1-args.tau_update) + args.tau_update * b_observations.mean(axis=0)
            obs_std = obs_std * (1-args.tau_update) + args.tau_update * b_observations.std(axis=0)
            b_observations =   torch.tensor((b_observations - obs_mean) / obs_std, device = device)   
            b_next_observations = torch.tensor((rb.next_observations[b_inds, b_inds_envs] - obs_mean) / obs_std, device = device)
            b_actions =  torch.tensor(rb.actions[b_inds, b_inds_envs], device = device) 
            b_rewards =  torch.tensor(rb.rewards[b_inds, b_inds_envs], device = device) 
            b_dones = torch.tensor(rb.dones[b_inds, b_inds_envs], device = device) 
            b_z = torch.tensor(rb.zs[b_inds, b_inds_envs], device = device) 
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(b_next_observations, b_z)
                qf1_next_target = qf1_target(b_next_observations, b_z, next_state_actions)
                qf2_next_target = qf2_target(b_next_observations, b_z, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                # rewards
                LSD_reward = ((discriminator_LSD(b_next_observations) - discriminator_LSD(b_observations)) * b_z).sum(dim = -1) 
                # print('LSD reward shape : ', LSD_reward.shape)
                intrinsic_reward = LSD_reward 
                intrinsic_reward = torch.clamp(intrinsic_reward, -5, 5)
                # intrinsic_reward = discriminator(b_observations).detach()
                if args.keep_extrinsic_reward:
                    b_rewards = intrinsic_reward.flatten() * args.coef_intrinsic  + b_rewards * args.coef_extrinsic
                else:
                    b_rewards = intrinsic_reward.flatten() * args.coef_intrinsic  
                # rewards = b_rewards.flatten() 
                next_q_value = b_rewards + (1 - b_dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(b_observations, b_z, b_actions).view(-1)
            # print('qf1_a_values', qf1_a_values.shape)
            qf2_a_values = qf2(b_observations, b_z, b_actions).view(-1)
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
                    pi, log_pi, _ = actor.get_action(b_observations, b_z)
                    qf1_pi = qf1(b_observations, b_z, pi)
                    qf2_pi = qf2(b_observations, b_z, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(b_observations, b_z)
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
                wandb.log({
                "losses/qf1_values": qf1_a_values.mean().item(), 
                "losses/qf2_values": qf2_a_values.mean().item(), 
                "losses/qf1_loss": qf1_loss.item(), 
                "losses/qf2_loss": qf2_loss.item(), 
                "losses/qf_loss": qf_loss.item() / 2.0, 
                "losses/actor_loss": actor_loss.item(), 
                "losses/alpha": alpha, 
                # metrics
                "metrics/rewards_mean": b_rewards.mean().item(), 
                "metrics/LSD_reward_mean": LSD_reward.mean().item(), 
                "metrics/LSD_reward_max": LSD_reward.max().item(), 
                "metrics/LSD_reward_min": LSD_reward.min().item(),
                # print("SPS:", int(global_step / (time.time() - start_time)))
                "charts/SPS": int(global_step / (time.time() - start_time)), 
                "losses/alpha_loss": alpha_loss.item() if args.autotune else 0.0, 
                }, step = global_step) if args.track else None

        if global_step % args.metric_freq == 0 : 
            # shannon_entropy_mu, coverage_mu = env_check.get_shanon_entropy_and_coverage_mu(rb.observations[fixed_idx_un].reshape(-1, *envs.single_observation_space.shape))
            wandb.log({
                "charts/coverage" : env_check.get_coverage(),
                "charts/shannon_entropy": env_check.shannon_entropy(),
                # "charts/coverage_mu" : coverage_mu,
                # "charts/shannon_entropy_mu": shannon_entropy_mu,
                }, step = global_step) if args.track else None

        if global_step % args.fig_frequency == 0  and global_step > args.learning_starts:
            if args.make_gif : 
                # print('size rho', size_rho)
                # print('max x rho', rb.observations[max(rb.pos if not rb.full else rb.buffer_size-size_rho, 0):rb.pos if not rb.full else rb.buffer_size][0][:,0].max())
                image = env_plot.gif(obs_un = rb.observations[np.random.randint(0, rb.pos if not rb.full else rb.buffer_size, 100_000)],
                                     obs = rb.observations[max(rb.pos-int(size_rho) if not rb.full else rb.buffer_size-int(size_rho), 0):rb.pos if not rb.full else rb.buffer_size], 
                                    device= device)
                send_matrix(wandb, image,  "gif", global_step) if args.track else None
            
    # FINAL LOGGING
    print(f"global_coverage={env_check.get_coverage()}, global_shannon_entropy={env_check.shannon_entropy()}, running_episodic_return={running_episodic_return}")
    envs.close()
    wandb.finish(quiet=True) if args.track else None