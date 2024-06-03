# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from dataclasses import dataclass
from distutils.util import strtobool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from stable_baselines3.common.buffers import ReplayBuffer
from src.utils.replay_buffer import ReplayBuffer
from envs.wenv import Wenv
from envs.config_env import config
from torch.utils.tensorboard import SummaryWriter
# animation 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import imageio

from src.utils.replay_buffer import ReplayBuffer
from src.utils.wandb_utils import send_video, send_matrix, send_dataset
from envs.wenv import Wenv
from envs.config_env import config
import tyro

# SPECIFIC IQL
from agent import IQL
from torch.utils.data import DataLoader, TensorDataset


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
    wandb_project_name: str = "contrastive_exploration"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    num_envs: int = 1
    """the number of parallel environments"""


    # replay buffer init
    init_replay_buffer: bool = True
    """if toggled, the replay buffer will be initialized"""
    env_id: str = "HalfCheetah-v3"
    """the id of the environment"""
    algo_name: str = "v1_ppo_lipshitz_adaptive_sampling"
    """the name of the algorithm"""
    seed: int = 0
    """seed of the experiment"""
    dataset_name: str = "dataset"
    """the name of the dataset"""
    learning_starts: int = 5e3
    """timestep to start learning"""


    # Algorithm specific arguments
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
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""


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



    

if __name__ == "__main__":
    args = tyro.cli(Args)
    # args.seed=np.random.randint(0,100)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}" if not args.init_replay_buffer else f"sac_{args.env_id}__init__{args.algo_name}__{args.seed}"
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


    # SETUP BATCH
    max_action = float(envs.single_action_space.high[0])
    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
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
            capacity= args.buffer_size, 
            observation_space= envs.single_observation_space,
            action_space= envs.single_action_space,
            device= device,
            run_init_path = None if not args.init_replay_buffer else f"{args.env_id}__{args.algo_name}__{args.seed}",
            project_name= args.wandb_project_name,
            name_dataset= args.dataset_name,
            num_envs= args.num_envs
            )

    start_time = time.time()
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
            wandb.log({"episodic_return": r_0/args.num_envs, "episodic_length": l_0/args.num_envs, "global_step": global_step})
            print(f"global_step={global_step}, episodic_return={r_0/args.num_envs}, episodic_length={l_0/args.num_envs}")
            # writer.add_scalar("charts/episodic_return", r_0/args.num_envs, global_step)
            # writer.add_scalar("charts/episodic_length", l_0/args.num_envs, global_step)
        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        for idx, done in enumerate(dones):
            if done:
                obs[idx],infos[idx] = envs.envs[idx].reset()

            
        # SAC: training.
        if (global_step > args.learning_starts) or args.init_replay_buffer:
            # q network training
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                # rewards
                rewards = torch.tensor(data.rewards, dtype=torch.float32, device=device).flatten()
                next_q_value = rewards+ (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).flatten()
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
                wandb.log({"actor_loss": actor_loss.item(), 
                            "qf1_loss": qf1_loss.item(), 
                            "qf2_loss": qf2_loss.item(), 
                            "qf_loss": qf_loss.item(), 
                            "alpha": alpha, 
                            "rewards": rewards.mean().item(), 
                            "min_reward": rewards.min().item(), 
                            "max_reward": rewards.max().item(), 
                            "SPS": int(global_step / (time.time() - start_time)), 
                            "global_step": global_step})
                if args.autotune:
                    wandb.log({"alpha_loss": alpha_loss.item()})

    envs.close()
