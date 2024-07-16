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
    env_id: str = "Maze-Ur-v0"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 1e3
    """timestep to start learning"""
    policy_lr: float = 5e-4
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
    sac_training_steps: int = 500
    """the number of training steps in each SAC training loop"""
    nb_rollouts_freq: int = 2
    """the frequency of logging the number of rollouts"""

    #  CLASSIFIER SPECIFIC
    classifier_lr: float = 1e-3
    """the learning rate of the classifier"""
    classifier_epochs: int = 1
    """the number of epochs to train the classifier"""
    classifier_batch_size: int = 128
    """the batch size of the classifier"""
    feature_extractor: bool = False
    """if toggled, a feature extractor will be used"""
    percentage_time: float = 0/4
    """the percentage of the time to use the classifier"""
    epsilon: float = 1e-3
    """the epsilon of the classifier"""
    lambda_init: float = 100.0 #50 in mazes
    """the lambda of the classifier"""
    bound_spectral: float = 1.0
    """the bound spectral of the classifier"""
    clip_lim: float = 1000.0
    """the clipping limit of the classifier"""
    adaptive_sampling: bool = False
    """if toggled, the sampling will be adaptive"""
    lip_cte: float = 1.0
    """the lip constant"""
    use_sigmoid: bool = False
    """if toggled, the sigmoid will be used"""
    # ALGO specific 
    beta_ratio: float = 1/16
    """the ratio of the beta"""
    rho_update_freq: int = 0
    """the frequency of updating rho"""


    # REWARD SPECIFIC
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
    metra_training_steps: int = 50
    """the number of training steps in each metra training loop"""
    nb_skills: int = 4
    """the number of skills"""
    


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        env = Wenv(env_id=env_id, xp_id=run_name, **config[env_id])
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env = gym.wrappers.ClipAction(env)
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
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + nb_skills, 256)
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


class Discriminator(torch.nn.Module):
    def __init__(self,  
                state_dim, 
                z_dim,
                env_name, 
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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # METRA SETUP 
    args.num_envs = args.nb_skills
    z = -1/(args.nb_skills-1)*torch.ones((args.nb_skills, args.nb_skills)).to(device) + (1+1/(args.nb_skills-1))*torch.eye(args.nb_skills).to(device)
    z = z/z.norm(dim=0)
    z_one_hot = torch.eye(args.nb_skills).to(device)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    # variables + initilization
    max_step = config[args.env_id]['kwargs']['max_episode_steps']
    actor = Actor(envs, args.nb_skills).to(device)
    qf1 = SoftQNetwork(envs, args.nb_skills).to(device)
    qf2 = SoftQNetwork(envs, args.nb_skills).to(device)
    qf1_target = SoftQNetwork(envs, args.nb_skills).to(device)
    qf2_target = SoftQNetwork(envs, args.nb_skills).to(device)
    discriminator = Discriminator(
        state_dim = np.array(envs.single_observation_space.shape).prod(),
        z_dim = args.nb_skills,
        env_name = args.env_id,
        device = device,
        lip_cte = args.lip_cte,
        lambda_init = args.lambda_metra)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.q_lr)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    # CLASSIFIER
    classifier = Classifier(envs.single_observation_space, 
                            env_max_steps=max_step,
                            device=device, 
                            n_agent=1, 
                            lipshitz=False,
                            feature_extractor=args.feature_extractor, 
                            lim_up = args.clip_lim,
                            lim_down = -args.clip_lim,
                            env_id=args.env_id, 
                            lipshitz_regu=True,
                            bound_spectral=args.bound_spectral,
                            lip_cte=args.lip_cte,
                            lambda_init=args.lambda_init,
                            epsilon=args.epsilon,
                            use_sigmoid=args.use_sigmoid
                            ).to(device)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=args.classifier_lr)
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
        optimize_memory_usage = False,
        n_envs = args.num_envs
    )
    # add z to replay buffer
    rb.zs = np.zeros((rb.buffer_size, rb.n_envs, args.nb_skills), dtype=np.float32)
    start_time = time.time()
    nb_rollouts = 0
    pos_rho = 0
    nb_epoch_rho = 0
    obs_rho_list = []
    next_obs_rho_list = []
    dones_rho_list = []
    z_rho_list = []
    count_episode = 0
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device),torch.Tensor(z).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
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
                print('NB EPISODES', count_episode)
        
        if True in terminations:
            rb.add(obs.copy(), real_next_obs.copy(), actions.copy(), rewards.copy(), terminations.copy(), infos.copy()) 
            rb.zs[rb.pos] = np.array(z)
            pos_rho += args.num_envs
        else:
            if bernoulli.rvs(args.beta_ratio):
                rb.add(obs.copy(), real_next_obs.copy(), actions.copy(), rewards.copy(), terminations.copy(), infos.copy())
                rb.zs[rb.pos] = np.array(z)
                pos_rho += args.num_envs
        # rho
        obs_rho_list.append(obs)
        next_obs_rho_list.append(real_next_obs)
        dones_rho_list.append(terminations)
        z_rho_list.append(z)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # if global_step > args.learning_starts:
        #     training_step = global_step
        # ALGO LOGIC: training.
        if nb_rollouts >= args.nb_rollouts_freq*args.nb_skills and  global_step>=args.learning_starts:
            pos_rho = max(min(pos_rho, rb.pos-16),4)
            add_pos = args.rho_update_freq*max_step*(args.beta_ratio)*args.nb_rollouts_freq * args.num_envs
            add_pos = 0 if rb.pos-(pos_rho+add_pos)<= 0 else add_pos
            print('pos_rho', pos_rho)
            print('rb.pos', rb.pos)
            print('add_pos', add_pos)
            # CLASSIFIER TRAINING
            # rho
            batch_obs_rho = torch.tensor(np.array(obs_rho_list), dtype=torch.float32).to(device).squeeze(axis=1).view(-1, *envs.single_observation_space.shape)
            batch_next_obs_rho = torch.tensor(np.array(next_obs_rho_list), dtype=torch.float32 ).to(device).squeeze(axis=1).view(-1, *envs.single_observation_space.shape)
            batch_dones_rho = torch.tensor(np.array(dones_rho_list)).to(device).view(-1)
            batch_z_rho = torch.tensor(np.array(z_rho_list)).to(device).view(-1, args.nb_skills)
            # un
            nb_sample_un = int(args.classifier_batch_size*(1-args.beta_ratio))
            nb_sample_rho = int(args.classifier_batch_size*args.beta_ratio)
            # classifier epoch 
            classifier_epochs = max((rb.pos // args.classifier_batch_size) * args.classifier_epochs, (batch_obs_rho.shape[0] // args.classifier_batch_size) * args.classifier_epochs)
            # classifier_epochs = (batch_obs_rho.shape[0] // args.classifier_batch_size) * args.classifier_epochs
            total_classification_loss = 0
            total_lipshitz_regu = 0
            for epoch in range(classifier_epochs):
                # CLASSIFIER TRAINING
                # mb rho
                # mb_rho_idx = np.random.choice(np.arange(batch_obs_rho.shape[0]-1), args.classifier_batch_size, p=prob.numpy())
                mb_rho_idx = np.random.randint(0, batch_obs_rho.shape[0], args.classifier_batch_size)
                mb_obs_rho = batch_obs_rho[mb_rho_idx].clone().detach().to(device)
                mb_next_obs_rho = batch_next_obs_rho[mb_rho_idx].clone().detach().to(device)
                mb_done_rho = batch_dones_rho[mb_rho_idx].to(torch.float32).clone().detach().to(device)
                mb_z_rho = batch_z_rho[mb_rho_idx].clone().detach().to(device)

                # mb un
                mb_un_inds = np.random.randint(0, rb.pos-(pos_rho+add_pos), args.classifier_batch_size)
                mb_un_inds_envs = np.random.randint(0, args.num_envs, args.classifier_batch_size)
                mb_obs_un = torch.tensor(rb.observations[mb_un_inds, mb_un_inds_envs]).to(device).squeeze(axis=1)
                mb_next_obs_un = torch.tensor(rb.next_observations[mb_un_inds, mb_un_inds_envs]).to(device).squeeze(axis=1)
                mb_done_un = torch.tensor(rb.dones[mb_un_inds, mb_un_inds_envs]).to(device)
                mb_z_un = torch.tensor(rb.zs[mb_un_inds, mb_un_inds_envs]).to(device)
                # classifier loss + lipshitz regularization
                loss, _ = classifier.lipshitz_loss_ppo(batch_q= mb_obs_rho, batch_p = mb_obs_un, 
                                                        q_batch_s =  mb_obs_rho, q_batch_next_s = mb_next_obs_rho, q_dones = mb_done_rho,
                                                        p_batch_s = mb_obs_un, p_batch_next_s = mb_next_obs_un, p_dones = mb_done_un, 
                                                        N_rho = batch_obs_rho.shape[0], N_un = rb.pos)       
                classifier_optimizer.zero_grad()
                loss.backward()
                classifier_optimizer.step()
                total_classification_loss += loss.item()/classifier_epochs
                # lambda loss
                _, lipshitz_regu = classifier.lipshitz_loss_ppo(batch_q= mb_obs_rho, batch_p = mb_obs_un, 
                                                        q_batch_s =  mb_obs_rho, q_batch_next_s = mb_next_obs_rho, q_dones = mb_done_rho,
                                                        p_batch_s = mb_obs_un, p_batch_next_s = mb_next_obs_un, p_dones = mb_done_un)    
                lambda_loss = classifier.lambda_lip*lipshitz_regu
                classifier_optimizer.zero_grad()
                lambda_loss.backward()
                classifier_optimizer.step()
                total_lipshitz_regu += lipshitz_regu.item()/classifier_epochs
                # METRA
                beta = args.beta_ratio
                # beta = 0.0
                loss =beta*discriminator.lipshitz_loss(mb_obs_rho, 
                                                   mb_next_obs_rho, 
                                                   mb_z_rho,
                                                   mb_done_rho) + (1-beta)*discriminator.lipshitz_loss(mb_obs_un,
                                                                                                mb_next_obs_un,
                                                                                                mb_z_un,
                                                                                                mb_done_un)
                discriminator_optimizer.zero_grad()
                loss.backward()
                discriminator_optimizer.step()
                # lambda loss
                lambda_loss =beta*discriminator.lambda_loss(mb_obs_rho, 
                                                   mb_next_obs_rho, 
                                                   mb_z_rho,
                                                   mb_done_rho) + (1-beta)*discriminator.lambda_loss(mb_obs_un,
                                                                                                mb_next_obs_un,
                                                                                                mb_z_un,
                                                                                                mb_done_un)
                discriminator_optimizer.zero_grad()
                lambda_loss.backward()
                discriminator_optimizer.step()


            # ALGO LOGIC: training.
            for training_step in range(args.sac_training_steps):
                batch_inds = np.random.randint(0, rb.pos, args.batch_size)
                batch_inds_envs = np.random.randint(0, args.num_envs, args.batch_size)
                observations = torch.tensor(rb.observations[batch_inds, batch_inds_envs]).to(device)
                next_observations = torch.tensor(rb.next_observations[batch_inds, batch_inds_envs]).to(device)
                actions = torch.tensor(rb.actions[batch_inds, batch_inds_envs]).to(device)
                rewards = torch.tensor(rb.rewards[batch_inds, batch_inds_envs]).to(device)
                dones = torch.tensor(rb.dones[batch_inds, batch_inds_envs]).to(device)
                zs = torch.tensor(rb.zs[batch_inds, batch_inds_envs]).to(device)
                # sample from replay buffer
                # observations = data.observations
                # next_observations = data.next_observations
                # actions = data.actions             
                # rewards = data.rewards   
                # dones = data.dones 
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(next_observations, zs)
                    qf1_next_target = qf1_target(next_observations, zs, next_state_actions)
                    qf2_next_target = qf2_target(next_observations, zs, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    # EXPLORATION TERM 
                    explore_reward = (classifier(next_observations).squeeze() - classifier(observations).squeeze())*1.0
                    # METRA TERM
                    phi_s = discriminator(observations)
                    phi_s_next = discriminator(next_observations)
                    metra_reward = ((phi_s_next - phi_s) * zs).sum(dim = -1)

                    intrinsic_reward = metra_reward + explore_reward*0.5
                    # intrinsic_reward = (intrinsic_reward - intrinsic_reward.mean()) / (intrinsic_reward.std() + 1e-6)
                    # intrinsic_reward = classifier(observations).squeeze()
                    # print('intrinsic_reward', intrinsic_reward.mean().item())
                    # intrinsic_reward += intrinsic_reward.min()
                    batch_rewards = args.coef_extrinsic * rewards.flatten() + args.coef_intrinsic * intrinsic_reward if args.keep_extrinsic_reward else args.coef_intrinsic * intrinsic_reward
                    next_q_value = batch_rewards + (1 - dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = qf1(observations, zs, actions).view(-1)
                qf2_a_values = qf2(observations, zs, actions).view(-1)
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
                        pi, log_pi, _ = actor.get_action(observations, zs)
                        qf1_pi = qf1(observations, zs, pi)
                        qf2_pi = qf2(observations, zs, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        # kl_actor 
                        # online
                        # mean, log_std = actor(observations)
                        # std = log_std.exp()
                        # normal = torch.distributions.Normal(mean, std)
                        # # target
                        # mean_target, log_std_target = actor(next_observations)
                        # std_target = log_std_target.exp()
                        # normal_target = torch.distributions.Normal(mean_target, std_target)
                        # # kl
                        # kl = torch.distributions.kl_divergence(normal, normal_target).mean()
                        # actor_loss += 0.5 * kl

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        if args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(observations, zs)
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
                    
                    # writer.add_scalar("stats/pos_rho", pos_rho, global_step)
                    # print("SPS:", int(training_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(training_step / (time.time() - start_time)), global_step)
                    if args.autotune:
                        writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
            # reinit
            nb_rollouts = 0
            pos_rho = 0
            obs_rho_list = []
            next_obs_rho_list = []
            dones_rho_list = []
            z_rho_list = []
            

        if global_step % args.fig_frequency == 0  and global_step > 0:
            if args.make_gif : 
                image = env_plot.gif(obs_un = rb.observations[:rb.pos],
                                    classifier = classifier,
                                    device= device)
                send_matrix(wandb, image, "gif", global_step)
            

    envs.close()
    writer.close()