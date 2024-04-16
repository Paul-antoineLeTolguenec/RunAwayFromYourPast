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
from src.utils.replay_buffer import ReplayBuffer
from src.ce.classifier import Classifier
from envs.wenv import Wenv
from envs.config_env import config
from torch.utils.tensorboard import SummaryWriter
# animation 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import imageio



def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_env", type=int, default=1)  
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
    parser.add_argument("--fig_frequency", type=int, default=1000)
    parser.add_argument("--make-gif", type=bool, default=True)
    parser.add_argument("--plotly", type=bool, default=False)
    parser.add_argument("--episodic-return", type=bool, default=True)
    parser.add_argument("--num-envs", type=int, default=1)

    # SAC
    parser.add_argument("--env-id", type=str, default="Swimmer-v3",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=int(1e6),
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=5000,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=5e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=2e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--q-frequency", type=int, default=1,
        help="the frequency of training Q network")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.2,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.2,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=False, 
        help="automatic tuning of the entropy coefficient")
    # CLASSIFIER
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-batch-size", type=int, default=256)
    parser.add_argument("--classifier-frequency", type=int, default=1)
    parser.add_argument("--classifier-per-add", type=float, default= 2/4)
    parser.add_argument("--classifier-capacity", type=int, default=1000)
    parser.add_argument("--ratio-reward", type=float, default=1.0)
    # parser.add_argument("--clip_reward", type=float, default=10.0)
    parser.add_argument("--feature-extractor", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--lipshitz", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--bound-spectral", type=float, default=1.0)

    # SEPCIFIC RHO 
    parser.add_argument("--pos_init_rho", type=int, default=5000, 
                        help="initial value of rho")
    parser.add_argument("--polyak", type=float, default=0.995)
    parser.add_argument("--nb_ep_rho", type=int, default=1)
    parser.add_argument("--sample_rho", type=lambda x: bool(strtobool(x)), default=False)

    args = parser.parse_args()
    # fmt: on
    return args


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
            "action_scale", torch.tensor((env.envs[0].action_space.high - env.envs[0].action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.envs[0].action_space.high + env.envs[0].action_space.low) / 2.0, dtype=torch.float32)
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
    
def select_rho(rho_pos, classifer, rb, device):
    # rho 
    rho_dist = rb.observations[rho_pos:rb.pos]
    with torch.no_grad():
        rho_pred = classifier(torch.tensor(rho_dist, dtype=torch.float32, device=device)).flatten()
    max_rho_pred = torch.max(rho_pred)
    mask_rho = rho_pred >= max_rho_pred * 0.5
    args_mask_rho = torch.arange(rho_pos, rb.pos, device=device)[mask_rho]
    # save 
    actions_ = rb.actions[args_mask_rho]
    rewards_ = rb.rewards[args_mask_rho]
    next_observations_ = rb.next_observations[args_mask_rho]
    dones_ = rb.dones[args_mask_rho]
    times_ = rb.times[args_mask_rho]
    # remove b_init_pos:b_end_pos from rb
    rb.observations = np.concatenate([np.delete(rb.observations, args_mask_rho, axis=0), np.zeros((len(args_mask_rho), rb.observation_space.shape[0]))], axis=0)
    rb.actions = np.concatenate([np.delete(rb.actions, args_mask_rho, axis=0) , np.zeros((len(args_mask_rho), rb.action_space.shape[0]))], axis=0)
    rb.rewards = np.concatenate([np.delete(rb.rewards, args_mask_rho, axis=0) , np.zeros((len(args_mask_rho), 1))], axis=0)
    rb.next_observations = np.concatenate([np.delete(rb.next_observations, args_mask_rho, axis=0) , np.zeros((len(args_mask_rho), rb.observation_space.shape[0]))], axis=0)
    rb.dones = np.concatenate([np.delete(rb.dones, args_mask_rho, axis=0) , np.zeros((len(args_mask_rho), 1))], axis=0)
    rb.times = np.concatenate([np.delete(rb.times, args_mask_rho, axis=0) , np.zeros((len(args_mask_rho), 1))], axis=0)
    rb.pos -= len(args_mask_rho)
    # replace to end of rb 
    rb.observations[rb.pos:rb.pos+len(args_mask_rho)] = rho_dist[mask_rho.cpu().numpy()]
    rb.actions[rb.pos:rb.pos+len(args_mask_rho)] = actions_
    rb.rewards[rb.pos:rb.pos+len(args_mask_rho)] = rewards_
    rb.next_observations[rb.pos:rb.pos+len(args_mask_rho)] = next_observations_
    rb.dones[rb.pos:rb.pos+len(args_mask_rho)] = dones_
    rb.times[rb.pos:rb.pos+len(args_mask_rho)] = times_
    # rho_pos update
    rb.pos += len(args_mask_rho)
    rho_pos = rb.pos - len(args_mask_rho)
    
    return rb, rho_pos

def update_probs(rb, rho_pos, classifier, device, batch_size = 256):
    un_indices = np.random.randint(0, rho_pos, size=batch_size)
    un_observations = torch.tensor(rb.observations[un_indices], dtype=torch.float32, device=device)
    with torch.no_grad():
        probs = torch.sigmoid(classifier(un_observations)).detach().cpu().numpy()
    rb.probs[un_indices] = probs
    return rb
    
if __name__ == "__main__":
    args = parse_args()
    # args.seed=np.random.randint(0,100)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"
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


    # SETUP BATCH 
    if args.episodic_return:
        max_steps = envs.envs[0].spec.max_episode_steps  


    max_action = float(envs.single_action_space.high[0])
    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    classifier = Classifier(envs.single_observation_space, 
                            env_max_steps=max_steps,
                            device=device, 
                            n_agent=1, 
                            lipshitz=args.lipshitz,
                            feature_extractor=args.feature_extractor, 
                            bound_spectral=args.bound_spectral,
                            env_id=args.env_id).to(device)
                            
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    classifier_optimizer = optim.Adam(list(classifier.parameters()), lr=args.classifier_lr)
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
        capacity = args.buffer_size,
        classifier_capacity = args.classifier_capacity,
        observation_space = envs.single_observation_space,
        action_space = envs.single_action_space,
        device = device,
        n_env = args.n_env
    )
    # INIT RB 
    rb.observations[:args.pos_init_rho] = np.repeat(envs.envs[0].reset()[0].reshape(1,-1), args.pos_init_rho, axis=0)
    rb.actions[:args.pos_init_rho] = np.array([envs.single_action_space.sample() for _ in range(args.pos_init_rho)])
    rb.rewards[:args.pos_init_rho] = np.zeros((args.pos_init_rho, 1))
    rb.next_observations[:args.pos_init_rho] = np.repeat(envs.envs[0].reset()[0].reshape(1,-1), args.pos_init_rho, axis=0)
    rb.dones[:args.pos_init_rho] = np.zeros((args.pos_init_rho, 1))
    rb.times[:args.pos_init_rho] = np.zeros((args.pos_init_rho, 1))
    rb.pos = args.pos_init_rho

    start_time = time.time()

    # INIT RHO VARIABLES
    pos_rho = args.pos_init_rho
    nb_ep_rho = 0
    mean_un = 0
    mean_un_1 = -10


    # TRY NOT TO MODIFY: start the game
    obs,infos = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, truncated, infos = envs.step(actions)
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if True in dones or True in truncated:
            r_0 = 0 
            l_0 = 0
            for info in infos['final_info']:
                r_0 += info['episode']['r']
                l_0 += info['episode']['l']
            writer.add_scalar("charts/episodic_return", r_0/args.n_env, global_step)
            writer.add_scalar("charts/episodic_length", l_0/args.n_env, global_step)
             # rho 
            # print('time', infos['final_info'][0]['episode']['l'])
            # max_time_rho, pos_max_time_rho = (infos[idx]['l'], rb.pos) if infos[idx]['l'] > max_time_rho else (max_time_rho, pos_max_time_rho)
            nb_ep_rho += 1
            if  mean_un_1 > 0 and nb_ep_rho > args.nb_ep_rho:
                # mean_un = max(mean_un_1,0)
                mean_un_1 = -10
                nb_ep_rho = 0
                rb, pos_rho = select_rho(pos_rho, classifier, rb, device)

            # print(f"Episodic return of the environment: {r_0/args.n_env}")
        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        # for idx, d in enumerate(dones):
        #     if d:
        #         real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        for idx, done in enumerate(dones):
            if done:
                obs[idx],infos[idx] = envs.envs[idx].reset()

        if global_step % args.classifier_frequency == 0 :
            print('pos_rho', pos_rho)
            print('self.pos', rb.pos)
            print('mean_un_1', mean_un_1)
            # classifier training
            batch_p, batch_q, un_indices, rho_indices = rb.sample_threshold(pos_rho, args.classifier_batch_size, args.classifier_per_add)
            classifier_loss, log_rho_un, prob_un= classifier.ce_loss_ppo(batch_q, batch_p, return_log_and_prob =True)
            classifier_optimizer.zero_grad()
            classifier_loss.backward()
            classifier_optimizer.step()
            # update mean_un_1
            mean_un_1 = mean_un_1*args.polyak + log_rho_un*(1-args.polyak)
            # update probs
            # rb = update_probs(rb, pos_rho, classifier, device)
        # SAC: training.
        if global_step > args.learning_starts:
            # q network training
            if global_step % args.q_frequency == 0:
                for _ in range(args.q_frequency):
                    data = rb.sample(args.batch_size, pos_rho=pos_rho) if args.sample_rho else rb.sample(args.batch_size)
                    with torch.no_grad():
                        next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                        qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                        qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                        # rewards produced by classifier
                        rewards = classifier(data.observations).detach().flatten()*args.ratio_reward
                        # normalize rewards
                        next_q_value = rewards+ (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                    qf1_a_values = qf1(data.observations, data.actions).view(-1)
                    qf2_a_values = qf2(data.observations, data.actions).view(-1)
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
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
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

            if global_step % 10 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("losses/rewards", rewards.mean().item(), global_step)
                writer.add_scalar("losses/min_reward", rewards.min().item(), global_step)
                writer.add_scalar("losses/max_reward", rewards.max().item(), global_step)
                writer.add_scalar("losses/classifier_loss", classifier_loss.item(), global_step)
                # print("SPS:", int(global_step / (time.time() - start_time)))
                # print(f"Global step: {global_step}")
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        if global_step % args.fig_frequency == 0 and args.make_gif and global_step > 0:
            sample_full = rb.full_rb_batch()
            obs_rho = sample_full.observations[pos_rho:]
            if args.make_gif : 
                env_plot.gif(obs_un = sample_full.observations, obs_un_train=None, obs= obs_rho, classifier=classifier, device=device)
            if args.plotly:
                env_plot.plotly(obs_un = sample_full.observations, obs_un_train=None, obs= None, classifier=classifier, device=device)
           

    envs.close()
    writer.close()