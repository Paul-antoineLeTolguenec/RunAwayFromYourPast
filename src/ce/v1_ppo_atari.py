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
from src.ce.classifier_atari import ClassifierAtari
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
    wandb_project_name: str = "cleanRL"
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

    # CLASSIFIER SPECIFIC
    classifier_lr: float = 1e-3
    """the learning rate of the classifier"""
    classifier_epochs: int = 32
    """the number of epochs to train the classifier"""
    classifier_batch_size: int = 256
    """the batch size of the classifier"""
    feature_extractor: bool = False
    """if toggled, a feature extractor will be used"""
    lipshitz: bool = True   
    """if toggled, the classifier will be Lipshitz"""
    bound_spectral: float = 1.0
    """the spectral bound of the classifier"""
    frac_wash: float = 1/4
    """the fraction of the dataset to wash"""
    percentage_time: float = 0/4
    """the percentage of the time to use the classifier"""
    n_agent: int = 1
    """the number of agents"""
    learn_z: bool = False
    """if toggled, the classifier will learn z"""
    use_lstm: bool = False
    """if toggled, the classifier will use LSTM"""
    lim_down: float = -10.0
    """the lower bound of the classifier"""
    lim_up: float = 10.0
    """the upper bound of the classifier"""

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
        env = MaxAndSkipEnv(env, skip=4)
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
    
def update_train(obs_un, obs_un_train, classifier, device, args):
    with torch.no_grad():
        n_batch = int(args.classifier_batch_size)
        n_replace = int(n_batch*args.frac_wash)
        idx_un = np.random.randint(0, obs_un.shape[0], size = n_batch)
        batch_un = obs_un[idx_un]
        idx_un_train = np.random.randint(0, obs_un_train.shape[0], size = n_batch)
        batch_un_train = obs_un_train[idx_un_train]
        # probs batch un
        batch_probs_un = (torch.sigmoid(classifier(torch.Tensor(batch_un).to(device)))).detach().cpu().numpy().squeeze(-1) 
        batch_probs_un_norm = batch_probs_un/batch_probs_un.sum()
        idx_replace = np.random.choice(idx_un, size=n_replace, p=batch_probs_un_norm)
        # probs un train
        probs_un_train = (1-torch.sigmoid(classifier(torch.Tensor(batch_un_train).to(device)))).detach().cpu().numpy().squeeze(-1) 
        probs_un_train_norm = probs_un_train/probs_un_train.sum()
        idx_remove = np.random.choice(idx_un_train, size=n_replace, p=probs_un_train_norm)
    obs_un_train[idx_remove] = obs_un[idx_replace].clone()
    return obs_un_train

def add_to_un(obs_un, 
              obs,
              obs_rho, 
              actions_rho, 
              logprobs_rho, 
              rewards_rho, 
              dones_rho, 
              values_rho, 
              times_rho, 
              dkl_rho_un,
              rate_dkl,
              classifier,
              args, 
              update):
    # if dkl_rho_un > 0 or True:
    if update > args.start_explore and dkl_rho_un >= 0 and rate_dkl >= 0:
        # KEEP BEST ROLLOUTS
        list_mean_rollouts = []
        with torch.no_grad():
            for i in range(len(obs_rho)):
                mean_rollout = torch.mean(classifier(obs_rho[i])).cpu().item()
                list_mean_rollouts.append(mean_rollout)
        ranked_rollouts = np.argsort(list_mean_rollouts)
        obs_un = torch.cat([obs_un, torch.cat([obs_rho[i].squeeze(1) for i in ranked_rollouts[:args.n_rollouts]],dim=0)], dim=0)
        # DELETE WORST ROLLOUTS FROM RHO
        for idx in sorted(ranked_rollouts[:args.n_rollouts], reverse=True):
            obs_rho.pop(idx)
            actions_rho.pop(idx)
            logprobs_rho.pop(idx)
            rewards_rho.pop(idx)
            dones_rho.pop(idx)
            values_rho.pop(idx)
            times_rho.pop(idx)
        # UPDATE DKL average
        dkl_rho_un = 0
        rate_dkl = 0
    return obs_un, obs_rho, actions_rho, logprobs_rho, rewards_rho, dones_rho, values_rho, times_rho, dkl_rho_un, rate_dkl

def select_best_rollout(obs_rho, classifier, args):
    list_mean_rollouts = []
    with torch.no_grad():
        for i in range(len(obs_rho)):
            mean_rollout = torch.mean(classifier(obs_rho[i])).detach().cpu().item()
            list_mean_rollouts.append(mean_rollout)
    ranked_rollouts = np.argsort(list_mean_rollouts)
    best_rollouts = ranked_rollouts[-args.n_best_rollout_to_keep:]
    return best_rollouts

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
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

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    classifier = ClassifierAtari(observation_space = envs.single_observation_space,
                device=device,
                lim_down = args.lim_down,
                lim_up = args.lim_up,
                learn_z = args.learn_z,
                n_agent = args.n_agent,
                env_id = args.env_id,
                feature_extractor = args.feature_extractor,
                use_lstm = args.use_lstm).to(device)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=args.classifier_lr, eps=1e-5)

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
    actions_rho = []
    logprobs_rho = []
    rewards_rho = []
    dones_rho = []
    values_rho = []
    times_rho = []
    # UN
    obs_un_train = torch.tensor(envs.envs[0].reset()[0], dtype=torch.float).unsqueeze(0).repeat(args.num_steps, 1, 1, 1)
    obs_un = obs_un_train.clone()
    # obs, infos = envs.reset(seed=args.seed)
    obs_un_plot = np.tile( np.array([[envs.envs[0].reset()[1]['position']['x'],envs.envs[0].reset()[1]['position']['y']]]) , (args.num_steps, 1))


    # INIT DKL_RHO_UN
    dkl_rho_un = args.mean_re_init
    last_dkl_rho_un = args.mean_re_init
    rate_dkl = 0

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
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
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
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        ######################################*** CLASSIFIER TRAINING ***######################################
        batch_obs_rho = obs.reshape(obs.shape[0]*obs.shape[1], obs.shape[2], obs.shape[3], obs.shape[4])
        batch_times_rho = times.reshape(times.shape[0]*times.shape[1]).unsqueeze(-1)
        max_time = batch_times_rho.max()  
        mask_rho = (batch_times_rho >= max_time * args.percentage_time).float().squeeze(-1)
        batch_obs_rho = batch_obs_rho[mask_rho.bool()]
        for epoch in range(args.classifier_epochs):
            mb_rho_idx = np.random.randint(0, batch_obs_rho.shape[0], args.classifier_batch_size)
            mb_rho = batch_obs_rho[mb_rho_idx]
            # mb_un_idx = np.random.randint(0, obs_un_train.shape[0], args.classifier_batch_size) if dkl_rho_un >= 0 and rate_dkl >= 0 else np.random.randint(0, obs_un.shape[0], args.classifier_batch_size)
            # mb_un = obs_un_train[mb_un_idx] if dkl_rho_un >= 0 and rate_dkl >= 0 else obs_un[mb_un_idx]
            mb_un_idx = np.random.randint(0, obs_un.shape[0], args.classifier_batch_size)
            mb_un = obs_un[mb_un_idx]
            # mb_un = obs_un[mb_un_idx]
            loss = classifier.ce_loss_ppo(mb_rho.to(device), mb_un.to(device))
            classifier_optimizer.zero_grad()
            loss.backward()
            classifier_optimizer.step()

        ######################################*** INTRINSIC REWARD ***######################################
        with torch.no_grad():
            log_rho_un = classifier(obs.reshape(obs.shape[0]*obs.shape[1], obs.shape[2], obs.shape[3], obs.shape[4]).to(device)).squeeze(-1).reshape(obs.shape[0], obs.shape[1])
        rewards = args.coef_intrinsic * log_rho_un + args.coef_extrinsic * rewards if args.keep_extrinsic_reward else args.coef_intrinsic * log_rho_un
        mask_pos = (log_rho_un > 0).float()
        ######################################*** UPDATE UN ***######################################
        # obs_un, obs_rho, actions_rho, logprobs_rho, rewards_rho, dones_rho, values_rho, times_rho, dkl_rho_un, rate_dkl = add_to_un(obs_un, 
        #                                                                                                                 obs, 
        #                                                                                                                 obs_rho, 
        #                                                                                                                 actions_rho, 
        #                                                                                                                 logprobs_rho, 
        #                                                                                                                 rewards, 
        #                                                                                                                 dones, 
        #                                                                                                                 values, 
        #                                                                                                                 times, 
        #                                                                                                                 dkl_rho_un, 
        #                                                                                                                 rate_dkl,
        #                                                                                                                 classifier,
        #                                                                                                                 args, 
        #                                                                                                                 iteration)
        ######################################*** UPDATE UN_TRAIN ***######################################

        ######################################*** ADD BEST ROLLOUT ***######################################

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
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()