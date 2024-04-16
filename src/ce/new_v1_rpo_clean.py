# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/rpo/#rpo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
# import specific 
from src.ce.classifier import Classifier
from envs.wenv import Wenv
from envs.config_env import config



@dataclass
class Args:
    # XP RECORD
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

    # GIF
    make_gif: bool = True
    """if toggled, will make gif """
    plotly: bool = False
    """if toggled, will use plotly instead of matplotlib"""
    fig_frequency: int = 1

    # RPO SPECIFIC
    env_id: str = "Maze-Ur"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    # """the number of parallel game environments"""
    # num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False #True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.05
    """coefficient of the entropy"""
    vf_coef: float = 1.0
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    rpo_alpha: float = 0.5
    """the alpha parameter for RPO"""

    # CLASSIFIER SPECIFIC
    classifier_lr: float = 1e-3
    """the learning rate of the classifier"""
    classifier_epochs: int = 16
    """the number of epochs to train the classifier"""
    classifier_batch_size: int = 256
    """the batch size of the classifier"""
    feature_extractor: bool = False
    """if toggled, a feature extractor will be used"""
    lipshitz: bool = True
    """if toggled, the classifier will be Lipshitz"""
    bound_spectral: float = 2.0
    """the spectral bound of the classifier"""
    frac_wash: float = 1/8
    """the fraction of the dataset to wash"""
    percentage_time: float = 2/4
    """the percentage of the time to use the classifier"""

    # RHO SPECIFIC
    n_best_rollout_to_keep: int = 4
    """the number of best rollouts to keep"""
    mean_re_init: float = -1.0
    """the mean re-init value"""
    polyak: float = 0.75
    """the polyak averaging coefficient"""
    n_rollouts: int = 4
    """the number of rollouts"""
    keep_extrinsic_reward: bool = False


    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        env = Wenv(env_id=env_id, xp_id=run_name, **config[env_id])
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env = gym.wrappers.ClipAction(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, rpo_alpha):
        super().__init__()
        self.rpo_alpha = rpo_alpha
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else:  # new to RPO
            # sample again to add stochasticity to the policy
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(device)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

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
        idx_replace = np.random.choice(idx_un, size=n_replace, p=batch_probs_un_norm, replace=False)
        # probs un train
        probs_un_train = (1-torch.sigmoid(classifier(torch.Tensor(batch_un_train).to(device)))).detach().cpu().numpy().squeeze(-1)
        probs_un_train_norm = probs_un_train/probs_un_train.sum()
        idx_remove = np.random.choice(idx_un_train, size=n_replace, p=probs_un_train_norm, replace=False)
    obs_un_train[idx_remove] = obs_un[idx_replace]
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
              classifier,
              args):
    if dkl_rho_un > 0 :
        # KEEP BEST ROLLOUTS
        list_mean_rollouts = []
        with torch.no_grad():
            for i in range(len(obs_rho)):
                mean_rollout = torch.mean(classifier(obs_rho[i])).cpu().item()
                list_mean_rollouts.append(mean_rollout)
        ranked_rollouts = np.argsort(list_mean_rollouts)
        best_rollouts = ranked_rollouts[-args.n_best_rollout_to_keep:]
        worst_rollouts = ranked_rollouts[:-args.n_best_rollout_to_keep]
        obs_un = torch.cat([obs_un, torch.cat([obs_rho[i].squeeze(1) for i in worst_rollouts],dim=0)], dim=0)
        # DELETE WORST ROLLOUTS FROM RHO
        obs_rho = [obs_rho[i] for i in best_rollouts]
        actions_rho = [actions_rho[i] for i in best_rollouts]
        logprobs_rho = [logprobs_rho[i] for i in best_rollouts]
        rewards_rho = [rewards_rho[i] for i in best_rollouts]
        dones_rho = [dones_rho[i] for i in best_rollouts]
        values_rho = [values_rho[i] for i in best_rollouts]
        times_rho = [times_rho[i] for i in best_rollouts]
        # UPDATE DKL average
        dkl_rho_un = args.mean_re_init
    elif obs_un is None: 
        obs_un = obs.clone().squeeze(1)
    return obs_un, obs_rho, actions_rho, logprobs_rho, rewards_rho, dones_rho, values_rho, times_rho, dkl_rho_un

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"
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
    # MAX STEPS
    max_steps = config[args.env_id]['kwargs']['max_episode_steps']
    args.num_steps = max_steps * args.n_rollouts
    # BATCH CALCULATION
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.classifier_epochs *= args.num_steps // args.classifier_batch_size
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
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    # AGENT
    agent = Agent(envs, args.rpo_alpha).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # CLASSIFIER
    classifier = Classifier(envs.single_observation_space, 
                            env_max_steps=max_steps,
                            device=device, 
                            n_agent=1, 
                            lipshitz=args.lipshitz,
                            feature_extractor=args.feature_extractor, 
                            bound_spectral=args.bound_spectral,
                            env_id=args.env_id)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=args.classifier_lr)

    # RPO: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    times = torch.zeros((args.num_steps, args.num_envs)).to(device)


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
    obs_un_train = torch.tensor(envs.envs[0].reset()[0], dtype=torch.float).to(device).unsqueeze(0).repeat(args.num_steps, 1)
    obs_un = None

    # INIT DKL_RHO_UN
    dkl_rho_un = args.mean_re_init

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    times[0] = torch.tensor(np.array([infos["l"]])).transpose(0,1).to(device)

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        # PLAYING IN ENV
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            times[step] = torch.tensor(np.array([infos["l"]])).transpose(0,1).to(device)
            done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        # UPDATE RHO 
        idx_dones = torch.where(dones)[0].cpu().numpy().tolist()
        idx_dones.append(obs.shape[0])
        idx_dones.insert(0,0)
        for i in range(len(idx_dones)-1):
            idx_start = idx_dones[i]
            idx_end = idx_dones[i+1]
            if obs[idx_start:idx_end].shape[0]!=0:
                obs_rho.append(obs[idx_start:idx_end])
                actions_rho.append(actions[idx_start:idx_end])
                logprobs_rho.append(logprobs[idx_start:idx_end])
                rewards_rho.append(rewards[idx_start:idx_end])
                dones_rho.append(dones[idx_start:idx_end])
                values_rho.append(values[idx_start:idx_end])
                times_rho.append(times[idx_start:idx_end])
        
        # CLASSIFIER TRAINING
        # batch_obs_rho = torch.cat(obs_rho, dim=0)
        # batch_times_rho = torch.cat(times_rho, dim=0)
        batch_obs_rho = obs.reshape(-1, obs.shape[-1])
        batch_times_rho = times.reshape(-1, times.shape[-1])
        max_time = batch_times_rho.max()  
        mask_rho = (batch_times_rho >= max_time * args.percentage_time).float().squeeze(-1)
        batch_obs_rho = batch_obs_rho[mask_rho.bool()]
        for epoch in range(args.classifier_epochs):
            mb_rho_idx = np.random.randint(0, batch_obs_rho.shape[0], args.classifier_batch_size)
            mb_rho = batch_obs_rho[mb_rho_idx]
            mb_un_idx = np.random.randint(0, obs_un_train.shape[0], args.classifier_batch_size)
            mb_un = obs_un_train[mb_un_idx]
            loss = classifier.ce_loss_ppo(batch_obs_rho, obs_un_train)
            classifier_optimizer.zero_grad()
            loss.backward()
            classifier_optimizer.step()

        # INTRINSIC REWARD
        with torch.no_grad():
            log_rho_un = classifier(obs)
        rewards = rewards + log_rho_un if args.keep_extrinsic_reward else log_rho_un
        # UPDATE DKL average
        dkl_rho_un = args.polyak * dkl_rho_un + (1-args.polyak) * log_rho_un.mean().item()
        print('DKL:', dkl_rho_un)
        # UPDATE UN
        obs_un, obs_rho, actions_rho, logprobs_rho, rewards_rho, dones_rho, values_rho, times_rho,dkl_rho_un = add_to_un(obs_un, 
                                                                                                                            obs,
                                                                                                                            obs_rho, 
                                                                                                                            actions_rho, 
                                                                                                                            logprobs_rho, 
                                                                                                                            rewards_rho, 
                                                                                                                            dones_rho, 
                                                                                                                            values_rho, 
                                                                                                                            times_rho, 
                                                                                                                            dkl_rho_un,
                                                                                                                            classifier,
                                                                                                                            args)
        # UPDATE TRAIN
        obs_un_train = update_train(obs_un, obs_un_train, classifier, device, args)
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
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
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
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

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        # explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if update % args.fig_frequency == 0  and global_step > 0:
            if args.make_gif : 
                env_plot.gif(obs_un, obs_un_train, obs.clone(), classifier, device)
            if args.plotly:
                env_plot.plotly(obs_un, obs_un_train, classifier, device)

    envs.close()
    writer.close()