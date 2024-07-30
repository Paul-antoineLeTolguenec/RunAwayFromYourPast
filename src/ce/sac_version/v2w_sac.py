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
from scipy.stats import bernoulli

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
    env_id: str = "Maze-Ur-v0"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(5e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 5e-4
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
    sac_training_steps: int = 200
    """the number of training steps in each SAC training loop"""
    nb_episodes_rho: int = 4
    """the number of episodes to keep in the rho"""

    #  discriminator SPECIFIC
    discriminator_lr: float = 1e-4
    """the learning rate of the discriminator"""
    lambda_lr: float = 1e-1
    """the learning rate of the lambda"""
    discriminator_epochs: int = 1
    """the number of epochs to train the discriminator"""
    discriminator_batch_size: int = 128
    """the batch size of the discriminator"""
    percentage_time: float = 0/4
    """the percentage of the time to use the discriminator"""
    epsilon: float = 1e-3
    """the epsilon of the discriminator"""
    lambda_init: float = 100.0 #50 in mazes
    """the lambda of the discriminator"""
    lip_cte: float = 1.0
    """the lip constant"""
    # ALGO specific 
    beta_ratio: float = 1/32
    """the ratio of the beta"""
    pad_rho: int = 8
    """the delta with actual rho"""

    # rewards
    keep_extrinsic_reward: bool = False
    """if toggled, the extrinsic reward will be kept"""
    coef_extrinsic: float = 1.0
    """the coefficient of the extrinsic reward"""
    coef_intrinsic: float = 1.0
    """the coefficient of the intrinsic reward"""

    # METRA SPECIFIC
    lambda_metra: float = 50.0
    """the lambda of the metra"""
    nb_skills: int = 4
    """the number of skills"""
    lr_metra_discriminator: float = 1e-4
    """the learning rate of the metra discriminator"""
    lr_lambda_metra: float = 1e-1
    """the learning rate of the lambda metra"""
    lip_cte_metra: float = 1.0
    """the constant of the lipschitz of the metra"""

    


def make_env(env_id, idx, capture_video, run_name, seed):
    def thunk():
        env = Wenv(env_id=env_id, xp_id=run_name, **config[env_id])
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed + idx)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, nb_skills):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape) + nb_skills, 256)
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
    def __init__(self, env, nb_skills):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + nb_skills, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc_mean = nn.Linear(1024, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(1024, np.prod(env.single_action_space.shape))
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

    def get_action(self, x, z):
        mean, log_std = self(x, z)
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
    
class METRA_Discriminator(torch.nn.Module):
    def __init__(self,  
                state_dim, 
                z_dim,
                env_name, 
                device, 
                lip_cte = 1.0,
                eps = 1e-6,
                lambda_init = 30.0):
        super(METRA_Discriminator, self).__init__()
        self.env_name = env_name
        self.l1=torch.nn.Linear(state_dim, 256).to(device)
        self.l2=torch.nn.Linear(256, 64).to(device)
        self.l3=torch.nn.Linear(64, z_dim).to(device)
        # learnable lagrange multiplier
        self.lambda_metra = nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32)) #lambda_metra in the paper
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
            save_code=False,
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
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # METRA SETUP 
    args.num_envs = args.nb_skills
    z = -1/(args.nb_skills-1)*torch.ones((args.nb_skills, args.nb_skills)).to(device) + (1+1/(args.nb_skills-1))*torch.eye(args.nb_skills).to(device)
    z = z/z.norm(dim=0)
    z_one_hot = torch.eye(args.nb_skills).to(device)
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.seed) for i in range(args.num_envs)]
    )
    for env in envs.envs: env.seed(args.seed)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    # variables + initilization
    max_step = config[args.env_id]['kwargs']['max_episode_steps']
    actor = Actor(envs, args.nb_skills).to(device)
    qf1 = SoftQNetwork(envs, args.nb_skills).to(device)
    qf2 = SoftQNetwork(envs, args.nb_skills).to(device)
    qf1_target = SoftQNetwork(envs, args.nb_skills).to(device)
    qf2_target = SoftQNetwork(envs, args.nb_skills).to(device)
    discriminator = Discriminator(envs, args.lambda_init, args.epsilon, args.lip_cte).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    # Discriminator
    discriminator = Discriminator(envs, args.lambda_init, args.epsilon, args.lip_cte).to(device)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.discriminator_lr)
    lambda_optimizer = optim.Adam([discriminator.lambda_param], lr=args.lambda_lr)
    metra_discriminator = METRA_Discriminator(state_dim = np.array(envs.single_observation_space.shape).prod(),
                                            z_dim = args.nb_skills,
                                            env_name = args.env_id,
                                            device = device,
                                            lip_cte = args.lip_cte_metra,
                                            eps = args.epsilon,
                                            lambda_init = args.lambda_metra).to(device)
    metra_optimizer = optim.Adam(list(metra_discriminator.parameters()), lr=args.lr_metra_discriminator)
    lambda_metra_optimizer = optim.Adam([metra_discriminator.lambda_metra], lr=args.lr_lambda_metra)
    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # specific un 
    size_un = max_step *  args.nb_episodes_rho
    fixed_idx_un = np.array([], dtype=int)
    # specific rho
    args.nb_episodes_rho = args.nb_episodes_rho*args.nb_skills
    print('args.nb_episodes_rho', args.nb_episodes_rho)
    size_rho = max_step * args.nb_episodes_rho 
    print('size_rho', size_rho)
    nb_rho_step = 0

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
        optimize_memory_usage = False, 
        n_envs=args.num_envs,
    )
    # add z to replay buffer
    rb.zs = np.zeros((rb.buffer_size, rb.n_envs, args.nb_skills), dtype=np.float32)
    start_time = time.time()
    nb_rollouts = 0
    pos_rho = 0
    nb_epoch_rho = 0
    count_episode = 0
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        nb_rho_step += args.num_envs
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            # actions = np.array([envs.envs[i].action_space.sample() for i in range(envs.num_envs)])
            actions = np.random.uniform(-max_action, max_action, size=(envs.num_envs, envs.single_action_space.shape[0]))

            # print('actions', actions) 
            # input() if 10<global_step else None
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device),torch.Tensor(z).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print('global_step', global_step)
                # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
                nb_rollouts += 1
                count_episode += 1
        
        
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        rb.zs[rb.pos-1] = z.cpu()
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

        # if global_step > args.learning_starts:
        #     training_step = global_step
        # ALGO LOGIC: training.
        if (nb_rollouts >= args.nb_episodes_rho or nb_rho_step >= size_rho) and global_step > args.learning_starts:
            print('Count episode', count_episode)
            print('nb_rho_step', nb_rho_step)
            print('nb_rollouts', nb_rollouts)
            print('size_rho', size_rho)
            print('rb.pos', rb.pos)
            # discriminator epoch 
            discriminator_epochs = (size_un // args.discriminator_batch_size) * args.discriminator_epochs
            total_classification_loss = 0
            total_lipshitz_regu = 0
            for epoch in range(discriminator_epochs):
                # discriminator TRAINING
                # batch un
                batch_inds_un = fixed_idx_un[np.random.randint(0, max(16,len(fixed_idx_un)-args.pad_rho * max_step * args.beta_ratio), args.batch_size)]
                batch_inds_envs_un = np.random.randint(0, args.num_envs, args.batch_size)
                observations_un = torch.Tensor(rb.observations[batch_inds_un, batch_inds_envs_un]).to(device)
                next_observations_un = torch.Tensor(rb.next_observations[batch_inds_un, batch_inds_envs_un]).to(device)
                rewards_un = torch.Tensor(rb.rewards[batch_inds_un, batch_inds_envs_un]).to(device)
                dones_un = torch.Tensor(rb.dones[batch_inds_un, batch_inds_envs_un]).to(device)
                z_un = torch.Tensor(rb.zs[batch_inds_un, batch_inds_envs_un]).to(device)

                # batch rho 
                batch_inds_rho = np.random.randint(rb.pos-size_rho, rb.pos, args.batch_size)
                batch_inds_envs_rho = np.random.randint(0, args.num_envs, args.batch_size)
                observations_rho = torch.Tensor(rb.observations[batch_inds_rho, batch_inds_envs_rho]).to(device)
                next_observations_rho = torch.Tensor(rb.next_observations[batch_inds_rho, batch_inds_envs_rho]).to(device)
                rewards_rho = torch.Tensor(rb.rewards[batch_inds_rho, batch_inds_envs_rho]).to(device)
                dones_rho = torch.Tensor(rb.dones[batch_inds_rho, batch_inds_envs_rho]).to(device)
                z_rho = torch.Tensor(rb.zs[batch_inds_rho, batch_inds_envs_rho]).to(device)
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
                total_lipshitz_regu += (constraints_rho.item() + constraints_un.item())/args.discriminator_epochs
                # train metra 
                beta = args.beta_ratio
                # beta = 0.0
                loss =beta*metra_discriminator.lipshitz_loss(observations_rho,
                                                    next_observations_rho,
                                                    z_rho,
                                                    dones_rho) + \
                                                    (1-beta)*metra_discriminator.lipshitz_loss(observations_un,
                                                                                    next_observations_un,
                                                                                    z_un,
                                                                                    dones_un)
                metra_optimizer.zero_grad()
                loss.backward()
                metra_optimizer.step()
                # lambda loss
                lambda_loss =beta*metra_discriminator.lambda_loss(observations_rho,
                                                    next_observations_rho,
                                                    z_rho,
                                                    dones_rho) + \
                                                    (1-beta)*metra_discriminator.lambda_loss(observations_un,
                                                                                    next_observations_un,
                                                                                    z_un,
                                                                                    dones_un)
                lambda_metra_optimizer.zero_grad()
                lambda_loss.backward()
                lambda_metra_optimizer.step()
                # clip lambda
                metra_discriminator.lambda_metra.data = torch.clamp(metra_discriminator.lambda_metra.data, 0, 1000)
                discriminator.lambda_param.data = torch.clamp(discriminator.lambda_param.data, 0, 1000)
            

            # ALGO LOGIC: training.
            for training_step in range(args.sac_training_steps):
                batch_inds_un = np.random.randint(0, rb.pos - int(size_rho/args.nb_skills), int(args.batch_size*(1-args.beta_ratio)))
                batch_inds_envs_un = np.random.randint(0, args.num_envs, int(args.batch_size*(1-args.beta_ratio)))
                # batch rho 
                batch_inds_rho = np.random.randint(rb.pos-size_rho, rb.pos, int(args.batch_size*args.beta_ratio))
                batch_inds_envs_rho = np.random.randint(0, args.num_envs, int(args.batch_size*args.beta_ratio))
                # data 
                b_observations = torch.cat([torch.Tensor(rb.observations[batch_inds_un, batch_inds_envs_un]).to(device), 
                                            torch.Tensor(rb.observations[batch_inds_rho, batch_inds_envs_rho]).to(device)], axis=0)
                b_next_observations = torch.cat([torch.Tensor(rb.next_observations[batch_inds_un, batch_inds_envs_un]).to(device),
                                            torch.Tensor(rb.next_observations[batch_inds_rho, batch_inds_envs_rho]).to(device)], axis=0)
                b_actions = torch.cat([torch.Tensor(rb.actions[batch_inds_un, batch_inds_envs_un]).to(device),
                                    torch.Tensor(rb.actions[batch_inds_rho, batch_inds_envs_rho]).to(device)], axis=0)
                b_rewards = torch.cat([torch.Tensor(rb.rewards[batch_inds_un, batch_inds_envs_un]).to(device),
                                    torch.Tensor(rb.rewards[batch_inds_rho, batch_inds_envs_rho]).to(device)], axis=0)
                b_dones = torch.cat([torch.Tensor(rb.dones[batch_inds_un, batch_inds_envs_un]).to(device),
                                    torch.Tensor(rb.dones[batch_inds_rho, batch_inds_envs_rho]).to(device)], axis=0)
                b_z = torch.cat([torch.Tensor(rb.zs[batch_inds_un, batch_inds_envs_un]).to(device),
                                    torch.Tensor(rb.zs[batch_inds_rho, batch_inds_envs_rho]).to(device)], axis=0)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(b_next_observations, b_z)
                    qf1_next_target = qf1_target(b_next_observations, b_z, next_state_actions)
                    qf2_next_target = qf2_target(b_next_observations, b_z, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    intrinsic_reward = (discriminator(b_next_observations).squeeze() - discriminator(b_observations).squeeze())
                    # clip intrinsic reward
                    # intrinsic_reward = torch.clamp(intrinsic_reward, -1, 1)
                    # metra intrinsic reward
                    phi_s = metra_discriminator(b_observations)
                    phi_s_next = metra_discriminator(b_next_observations)
                    metra_reward = ((phi_s_next - phi_s) * b_z).sum(dim = -1)
                    batch_rewards = metra_reward*2.0
                    # intrinsic_reward = (intrinsic_reward - intrinsic_reward.mean()) / (intrinsic_reward.std() + 1e-6)
                    # intrinsic_reward = discriminator(observations).squeeze()
                    # print('intrinsic_reward', intrinsic_reward.mean().item())
                    # intrinsic_reward += intrinsic_reward.min()
                    # batch_rewards = args.coef_extrinsic * rewards.flatten() + args.coef_intrinsic * intrinsic_reward if args.keep_extrinsic_reward else args.coef_intrinsic * intrinsic_reward
                    next_q_value = batch_rewards + (1 - b_dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = qf1(b_observations, b_z, b_actions).view(-1)
                qf2_a_values = qf2(b_observations, b_z, b_actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if training_step % args.policy_frequency == 0:  # TD 3 Delayed update support
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
                if training_step % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                if training_step % 10 == 0:
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/policy_entropy", -(torch.exp(log_pi) * log_pi).mean().item(), global_step)
                    # writer.add_scalar("losses/kl", kl.item(), global_step)
                    writer.add_scalar("losses/alpha", alpha, global_step)
                    writer.add_scalar("losses/total_classification_loss", total_classification_loss, global_step)
                    writer.add_scalar("losses/total_lipshitz_regu", total_lipshitz_regu, global_step)
                    writer.add_scalar("stats/nb_rollouts", nb_rollouts, global_step)
                    writer.add_scalar("stats/intrinsic_reward", intrinsic_reward.mean().item(), global_step)
                    writer.add_scalar("stats/intrinsic_reward_max", intrinsic_reward.max().item(), global_step)
                    writer.add_scalar("stats/intrinsic_reward_min", intrinsic_reward.min().item(), global_step)
                    writer.add_scalar("stats/metra_reward", metra_reward.mean().item(), global_step)
                    writer.add_scalar("stats/metra_reward_max", metra_reward.max().item(), global_step)
                    writer.add_scalar("stats/metra_reward_min", metra_reward.min().item(), global_step)
                    # metrics 
                    writer.add_scalar("metrics/lambda_discriminator", discriminator.lambda_param.item(), global_step)
                    writer.add_scalar("metrics/lambda_metra", metra_discriminator.lambda_metra.item(), global_step)
                    
                    # writer.add_scalar("stats/pos_rho", pos_rho, global_step)
                    # print("SPS:", int(training_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(training_step / (time.time() - start_time)), global_step)
                    if args.autotune:
                        writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

            # reinit
            nb_rollouts = 0
            nb_rho_step = 0
                
            

        if global_step % args.fig_frequency == 0  and global_step > 0:
            if args.make_gif : 
                image = env_plot.gif(obs_un = rb.observations[fixed_idx_un],
                                     obs = rb.observations[rb.pos-size_rho:rb.pos],
                                    classifier = discriminator,
                                    device= device)
                send_matrix(wandb, image, "gif", global_step)
            

    envs.close()
    writer.close()