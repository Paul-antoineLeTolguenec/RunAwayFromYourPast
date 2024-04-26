# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
# import specific 
import torch.nn.functional as F
from envs.wenv import Wenv
from envs.config_env import config

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
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

    # Algorithm specific arguments
    env_id: str = "PitfallNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # ICM SPECIFIC
    icm_lr: float = 1e-3
    """the learning rate of the intrinsic curiosity module"""
    beta: float = 0.2
    """the beta parameter for the intrinsic curiosity module"""
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
    """the coefficient of the extrinsic reward"""
    icm_epochs: int = 16

    # RHO SPECIFIC
    n_best_rollout_to_keep: int = 0
    """the number of best rollouts to keep"""
    mean_re_init: float = -10.0
    """the mean re-init value"""
    polyak: float = 0.75
    """the polyak averaging coefficient"""
    polyak_speed: float = 0.75
    """ polyak averagieng coefficient for speed """
    n_rollouts: int = 2
    """the number of rollouts"""
    keep_extrinsic_reward: bool = False
    """if toggled, the extrinsic reward will be kept"""
    start_explore: int = 4
    """the number of updates to start exploring"""
    coef_intrinsic: float = 1.0
    """the coefficient of the intrinsic reward"""
    coef_extrinsic: float = 1.0
    """the coefficient of the extrinsic reward"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = Wenv(env_id, **config[env_id])
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = NoopResetEnv(env, noop_max=10)
        # env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        # if "FIRE" in env.unwrapped.get_action_meanings():
        #     env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 1)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ICM(nn.Module):   
    def __init__(self, state_dim, action_dim,feature_dim=16, beta = 0.2, device='cpu'):
        super(ICM, self).__init__()
        # Intrinsic Curiosity Module
        # feature encoder
        self.cnn = nn.Sequential(
                            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
                            nn.ReLU(),
                            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                            nn.ReLU(),
                            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                            nn.ReLU(),
                            nn.Flatten(),
                            layer_init(nn.Linear(64 * 7 * 7, 512)),
                            nn.ReLU(),
                            ).to(device)
        self.flatten = torch.nn.Linear(512, feature_dim,device=device)
        # inverse model
        self.inverse_1 = nn.Linear(2*feature_dim, 32, device = device)
        self.inverse_2 = nn.Linear(32, action_dim, device = device)
        # forward model
        self.forward_1 = nn.Linear(feature_dim+1, 32, device = device)
        self.forward_2 = nn.Linear(32, feature_dim, device = device)
        # beta 
        self.beta = beta

    def feature(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        return x
    
    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.forward_1(x))
        x = self.forward_2(x)
        return x
    
    def inverse(self, x, next_x):
        x = torch.cat([x, next_x], 1)
        x = F.relu(self.inverse_1(x))
        x = self.inverse_2(x)
        return torch.softmax(x, dim=1)

    def loss(self, obs, next_obs, action, dones, reduce=True):
        
        # feature encoding
        phi = self.feature(obs)
        next_phi = self.feature(next_obs)
        # inverse model
        pred_a = self.inverse(phi, next_phi) * (1-dones)

        # forward model
        pred_phi = self.forward(phi, action.unsqueeze(-1)) * (1-dones)
        # losses
        # inverse_loss : MSE for continuous action space
        inverse_loss = torch.log(torch.gather(pred_a, 1, action.unsqueeze(-1).long())+1e-1).mean()
        # forward_loss : MSE for continuous action space
        forward_loss = torch.sqrt(F.mse_loss(pred_phi, next_phi, reduction='none').sum(1))
        forward_loss = torch.sqrt(forward_loss + 1e-6)
        return ((1-self.beta)*inverse_loss + self.beta*forward_loss.mean(), inverse_loss, forward_loss.mean()) if reduce else forward_loss

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(512, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x / 255.0)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state
    


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"


    # coverage check env 
    env_check = Wenv(env_id=args.env_id,
                    render_bool_matplot=False,
                    xp_id=run_name,
                    **config[args.env_id])

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    icm = ICM(state_dim=envs.single_observation_space.shape[0], 
              action_dim=envs.single_action_space.n, 
              device=device).to(device)
    optimizer_icm = optim.Adam(icm.parameters(), lr=args.icm_lr, eps=1e-5)

    # PPO Storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    times = torch.zeros((args.num_steps, args.num_envs)).to(device)
    positions = torch.zeros((args.num_steps, args.num_envs, 2))

    # RHO and UN: Storage setup
    # RHO 
    obs_rho = []
    obs_rho_not_terminated = [ [] for _ in range(args.num_envs)]
    actions_rho = []
    actions_rho_not_terminated = [ [] for _ in range(args.num_envs)]
    logprobs_rho = []
    logprobs_rho_not_terminated = [ [] for _ in range(args.num_envs)]
    rewards_rho = []
    rewards_rho_not_terminated = [ [] for _ in range(args.num_envs)]
    dones_rho = []
    dones_rho_not_terminated = [ [] for _ in range(args.num_envs)]
    values_rho = []
    values_rho_not_terminated = [ [] for _ in range(args.num_envs)]
    times_rho = []
    times_rho_not_terminated = [ [] for _ in range(args.num_envs)]
    nb_rollouts_per_episode = []

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
    )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
    times[0] = torch.tensor(np.array([infos["l"]])).to(device)
    for iteration in range(1, args.num_iterations + 1):
        ######################################*** SAMPLING ENV ***######################################
    
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        
        for step in range(0, args.num_steps):
            # coverage assessment
            env_check.update_coverage(obs = next_obs.cpu().numpy(), infos = infos)
        # while not np.all(terminations_episode):
            global_step += args.num_envs
            obs[step] = next_obs
            positions[step] = torch.tensor(np.array([[infos['position'][i]['x'],infos['position'][i]['y']] for i in range(args.num_envs)])).to(device)
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            times[step] = torch.tensor(np.array([infos["l"]])).to(device)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        print(f"global_step={global_step}, episodic_length={info['episode']['l']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                

        ######################################*** MODULE TRAINING ***######################################
        b_batch_obs = obs[:-1].reshape(-1, *envs.single_observation_space.shape)
        b_batch_next_obs = obs[1:].reshape(-1, *envs.single_observation_space.shape)
        b_batch_actions = actions[:-1].reshape(-1, *envs.single_action_space.shape)
        b_batch_dones = dones[:-1].reshape(-1, 1)

        for icm_update in range(args.icm_epochs):
            idx_mb_obs = torch.randint(0, b_batch_obs.shape[0]-1, (args.minibatch_size,))
            idx_mb_next_obs = idx_mb_obs + 1
            mb_obs = b_batch_obs[idx_mb_obs]
            mb_next_obs = b_batch_next_obs[idx_mb_next_obs]
            mb_actions = b_batch_actions[idx_mb_obs]
            mb_dones = b_batch_dones[idx_mb_obs]
            icm_loss, inverse_loss, forward_loss = icm.loss( mb_obs, mb_next_obs, mb_actions, mb_dones, reduce=True)
            optimizer_icm.zero_grad()
            icm_loss.backward()
            optimizer_icm.step()

        ######################################*** INTRINSIC REWARD ***######################################
        with torch.no_grad():
            # update the reward
            rewards_intrinsic = icm.loss(obs[:-1].reshape(-1, *envs.single_observation_space.shape), 
                                obs[1:].reshape(-1, *envs.single_observation_space.shape), 
                                actions[:-1].reshape(-1, *envs.single_action_space.shape), 
                                dones[:-1].reshape(-1, 1),
                                reduce=False).unsqueeze(-1)
            rewards_intrinsic = torch.cat([rewards_intrinsic, torch.zeros(args.num_envs,1).to(device)], 0).reshape(args.num_steps, args.num_envs)
            

        rewards = args.coef_extrinsic * rewards + args.coef_intrinsic*rewards_intrinsic  if args.keep_extrinsic_reward else args.coef_intrinsic * rewards_intrinsic



        ######################################*** PPO TRAINING ***######################################
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs,
                next_lstm_state,
                next_done,
            ).reshape(1, -1)
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
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

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

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
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
        writer.add_scalar("charts/coverage", env_check.get_coverage(), global_step)
        writer.add_scalar("charts/rooms", env_check.get_rooms(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()