# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from stable_baselines3.common.buffers import ReplayBuffer
from src.utils.replay_buffer_n import ReplayBuffer_n
from src.utils.classifier import Classifier
from src.utils.vae import VAE
from src.ce.vector_encoding import VE
from envs.continuous_maze import Maze
from torch.utils.tensorboard import SummaryWriter
# animation 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import imageio



def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-type", type=str, default="Maze")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=0,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="contrastive_exploration",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--do_fig", type=bool, default=True)
    parser.add_argument("--fig_frequency", type=int, default=1000)
    parser.add_argument("--make-gif", type=bool, default=True)

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Ur",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=int(1e6),
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.01,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=2**13,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=5e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.1,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--n-agent", type=int, default=5)
    parser.add_argument("--ratio-reward", type= float, default=1.0)
    parser.add_argument("--polyak-p", type=float, default=0.005)
    parser.add_argument("--auto-enco-lr", type=float, default=5e-4)
    args = parser.parse_args()
    args.n_envs = args.n_agent
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name, env_type = 'gym'):
    def thunk():
        env = gym.make(env_id) if env_type == 'gym' else Maze(name = env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, n_agent):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape) + 1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a, z):
        x = torch.cat([x, a, z], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, n_agent):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + 1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.envs[0].action_space.high - env.envs[0].action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.envs[0].action_space.high + env.envs[0].action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x, z):
        x = torch.cat([x, z], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, z):
        mean, log_std = self(x,z)
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


if __name__ == "__main__":
    args = parse_args()
    # args.seed=np.random.randint(0,100)
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

    if args.do_fig:
        # create folder if not exist
        if not os.path.exists('fig'):
            os.makedirs('fig')
        # env to plot 
        env_plot = Maze(name = args.env_id, fig = True)
        # iter_plot 
        iter_plot = 0
    if args.make_gif:
        if not os.path.exists('gif'):
            os.makedirs('gif')
        writer_gif = imageio.get_writer('gif/ssm.mp4', fps=2)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"using device {device}")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name, args.env_type) for _ in range(args.n_agent)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs,args.n_agent).to(device)
    qf1 = SoftQNetwork(envs,args.n_agent).to(device)
    qf2 = SoftQNetwork(envs,args.n_agent).to(device)
    qf1_target = SoftQNetwork(envs,args.n_agent).to(device)
    qf2_target = SoftQNetwork(envs,args.n_agent).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    classifier = Classifier(envs.single_observation_space.shape[0],device=device, n_agent=args.n_agent)
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    VAE = VAE(envs.single_observation_space.shape[0], 32, 32).to(device)
    VAE_optimizer = optim.Adam(VAE.parameters(), lr=args.auto_enco_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    classifier_optimizer = optim.Adam(list(classifier.parameters()), lr=args.classifier_lr)
    # VE 
    ve = VE(args.n_agent, device, torch.ones(args.n_agent)/args.n_agent)
    # generate n_agent different colors for matplotlib
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer_n(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        args.n_agent,
        torch.ones(args.n_agent)/args.n_agent,
        handle_timeout_termination=True,
    )
    start_time = time.time()
    batch_r_mean = 0
    running_mean_log_p = torch.zeros((1,args.n_agent), device=device)
    iter_plot = rb.pos.copy()
    # TRY NOT TO MODIFY: start the game
    obs,infos = envs.reset()
    # sample z 
    z_idx = torch.arange(1,args.n_agent+1)
    add_pos, increment_i = rb.incr_add_pos(z_idx)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device), z_idx.unsqueeze(1).to(device))
                actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, truncated, infos = envs.step(actions)
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if True in dones:
            r_0 = 0 
            l_0 = 0
            for info in infos['final_info']:
                r_0 += info['episode']['r']
                l_0 += info['episode']['l']
            writer.add_scalar("charts/episodic_return", r_0/args.n_agent, global_step)
            writer.add_scalar("charts/episodic_length", l_0/args.n_agent, global_step)
            print(f"Episodic return of the environment: {r_0/args.n_agent}")
        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        # for idx, d in enumerate(dones):
        #     if d:
        #         real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos, z_idx, increment_i, add_pos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        if True in dones:
            for idx, done in enumerate(dones):
                if done:
                    obs[idx],infos[idx] = envs.envs[idx].reset()
            #         z_idx[idx] = ve.sample(1)
            # add_pos, increment_i = rb.incr_add_pos(z_idx[idx])
        # ALGO LOGIC: training.
        # training if each element of rb.pos is greater than args.learning_starts
        if np.all(rb.pos > args.learning_starts):
            # sample batch 
            data = rb.sample(args.batch_size, ve)
            # classifier training
            loss_discriminator = classifier.mlh_loss(data.observations, data.z) 
            classifier_loss = loss_discriminator
            classifier_optimizer.zero_grad()
            classifier_loss.backward()
            classifier_optimizer.step()
            # VAE training
            loss_vae = VAE.loss(data.observations)
            print(f"loss_vae: {loss_vae}")
            VAE_optimizer.zero_grad()
            loss_vae.backward()
            VAE_optimizer.step()
            # update z
            true_z = data.z 
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations, true_z)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions, true_z)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions, true_z)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                # rewards produced by vae 
                log_p_s = VAE.loss_n(data.observations)
                # rewards produced by classifier
                log_p_z = torch.log(args.n_agent*torch.gather(classifier.softmax(classifier(data.observations)), 1, true_z.type(torch.int64)-1)).flatten()
                # rewards
                rewards = log_p_s + log_p_z
                next_q_value = rewards+ (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
            qf1_a_values = qf1(data.observations, data.actions, true_z).view(-1)
            qf2_a_values = qf2(data.observations, data.actions, true_z).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations, true_z)
                    qf1_pi = qf1(data.observations, pi, true_z)
                    qf2_pi = qf2(data.observations, pi, true_z)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations, data.z)
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

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
                writer.add_scalar("losses/r/rewards", rewards.mean().item(), global_step)
                writer.add_scalar("losses/r/min_reward", rewards.min().item(), global_step)
                writer.add_scalar("losses/r/max_reward", rewards.max().item(), global_step)
                # writer.add_scalar("losses/mean_no", mean_no.mean().item(), global_step)
                # writer.add_scalar("losses/std_no", std_no.mean().item(), global_step)
                writer.add_scalar("losses/classifier_loss", classifier_loss.item(), global_step)
                writer.add_scalar("losses/loss_discriminator", loss_discriminator.item(), global_step)

                # print("SPS:", int(global_step / (time.time() - start_time)))
                print(f"Global step: {global_step}")
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        if global_step % args.fig_frequency == 0 and args.do_fig and global_step > 0:
            # # remove only previous plot
            # env_plot.ax.clear()
            # # reset lim env plot 
            # env_plot.reset_lim_fig()
            for z in ve.z:
                idx_plot = z.cpu().numpy()-1
                if rb.pos[idx_plot] > 0:
                    x = rb.observations[idx_plot, iter_plot[idx_plot]:rb.pos[idx_plot],0]
                    y = rb.observations[idx_plot, iter_plot[idx_plot]:rb.pos[idx_plot],1]
                    env_plot.ax.scatter(x, y, c=colors[idx_plot], label=f'z={z}',s=1, alpha=0.5)
    
            # save fig env_plot
            env_plot.figure.canvas.draw()
            image = np.frombuffer(env_plot.figure.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(env_plot.figure.canvas.get_width_height()[::-1] + (3,))
            writer_gif.append_data(image)
            # update iter_plot
            iter_plot = rb.pos.copy()

    envs.close()
    writer.close()
