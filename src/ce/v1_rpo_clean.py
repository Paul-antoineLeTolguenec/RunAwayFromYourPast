# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

# import gymnasium as gym
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from src.utils.custom_sampling import exp_dec
from torch.utils.tensorboard import SummaryWriter
# from stable_baselines3.common.buffers import ReplayBuffer
from src.ce.classifier import Classifier
from envs.wenv import Wenv
from envs.config_env import config
# animation 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import imageio


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=0,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default= False, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="contrastive_exploration",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--make-gif", type=bool, default=True)
    parser.add_argument("--plotly", type=bool, default=False)
    parser.add_argument("--fig_frequency", type=int, default=1)


    parser.add_argument("--episodic-return", type=bool, default=True)

    # PPO
    parser.add_argument("--env-id", type=str, default="Hopper-v3",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--n-capacity", type=int, default=10**4,
        help="the capacity of the replay buffer in terms of episodes")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--num_rollouts", type=int, default=4,
        help="the number of rollouts ")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--minibatch-size", type=int, default=128,
                        help="the size of the mini-batch")
    parser.add_argument("--update-epochs", type=int, default= 16,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-coef-mask", type=float, default=0.4,
        help="the surrogate clipping coefficient for mask")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    # parser.add_argument("--ent-coef-mask", type=float, default=0.1,
    #     help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=1.0,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--rpo-alpha", type=float, default=0.1)
    # CLASIFIER
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-batch-size", type=int, default=256)
    parser.add_argument("--classifier-memory", type=int, default=1000)
    parser.add_argument("--classifier-frequency", type=int, default=1)
    parser.add_argument("--classifier-epochs", type=int, default=8)
    parser.add_argument("--feature-extractor", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--lipshitz", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--bound-spectral", type=float, default=1)
    parser.add_argument("--frac-wash", type=float, default=1/8, help="fraction of the buffer to wash")
    parser.add_argument("--start-explore", type=int, default=1)
    parser.add_argument("--ratio-reward", type=float, default=1.0)
    parser.add_argument("--treshold-entropy", type=float, default=0.0)
    parser.add_argument("--treshold-success", type=float, default=0.0)
    parser.add_argument("--per-threshold", type=float, default=2/4)
    parser.add_argument("--per-max-step", type=float, default=2/4)
    parser.add_argument("--nb_success", type=int, default=8)
    parser.add_argument("--update-un-frequency", type=int, default=1)
    parser.add_argument("--ratio-speed", type=float, default=1.0)
    args = parser.parse_args()
    # args.num_steps = args.num_steps // args.num_envs
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CountSuccess:
    def __init__(self, n_success):
        self.n_success = n_success
        self.success = 0
    def update(self, success_epoch):
        self.success = self.success + 1 if success_epoch  else 0
    def get(self):
        if self.success >= self.n_success : 
            self.success = 0
            return True
        else:
            return False
    

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

    def get_action_and_value(self, x, action=None, classifier = None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std) 
        if action is None:
            # probs = probs if classifier(x) > 0 else Normal(action_mean, action_std/100.0)
            action = probs.sample()
        else:  # new to RPO
            # sample again to add stochasticity to the policy
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(device)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

def update_train(obs_un, obs_un_train, frac,capacity, 
                 n_past, n_rollouts, max_steps, 
                 n_envs, classifier, device):
    # n_batch = int(obs_un_train.shape[0])
    n_batch = capacity
    n_replace = int(obs_un_train.shape[0])
    idx = np.random.randint(0, max(obs_un.shape[0]-max_steps*n_rollouts*n_past*n_envs,1), size = n_batch)
    batch_obs_un = obs_un[idx]
    # probs batch un
    batch_probs_un = (torch.sigmoid(classifier(torch.Tensor(batch_obs_un).to(device)))).detach().cpu().numpy().squeeze(-1) 
    batch_probs_un_norm = batch_probs_un/batch_probs_un.sum()
    idx_replace = np.random.choice(batch_obs_un.shape[0], n_replace, p=batch_probs_un_norm)
    # idx_replace = np.random.randint(0, obs_un_train.shape[0], size = n_replace)
    # probs un train
    probs_un_train = (1-torch.sigmoid(classifier(torch.Tensor(obs_un_train).to(device)))).detach().cpu().numpy().squeeze(-1) 
    probs_un_train_norm = probs_un_train/probs_un_train.sum()
    idx_remove = np.random.choice(obs_un_train.shape[0], n_replace, p=probs_un_train_norm)
    # idx_remove = np.random.randint(0, obs_un_train.shape[0], size = n_replace)
    # replace
    obs_un_train[idx_remove] = batch_obs_un[idx_replace]
    return obs_un_train

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            # monitor_gym=True, no longer works for gymnasium
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
    if args.episodic_return:
        max_steps = envs.envs[0].spec.max_episode_steps  
        args.num_steps = max_steps * args.num_rollouts
        args.classifier_memory = max_steps*args.num_rollouts*args.num_envs 
    # update batch size and minibatch size
    args.batch_size = int(args.num_envs * args.num_steps)
    # print('batch_size',args.batch_size)
    args.num_mini_batch = args.batch_size // args.minibatch_size
    
   
    # Agent
    agent = Agent(envs, args.rpo_alpha).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    classifier = Classifier(envs.single_observation_space, 
                            env_max_steps=max_steps,device=device, 
                            n_agent=1, 
                            lipshitz=args.lipshitz,
                            feature_extractor=args.feature_extractor, 
                            bound_spectral=args.bound_spectral,
                            env_id=args.env_id).to(device)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=args.classifier_lr, eps=1e-5)
    cs = CountSuccess(args.nb_success)
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    times = torch.zeros((args.num_steps, args.num_envs) + (1,)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # backup
    obs_backup = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions_backup = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    times_backup = torch.zeros((args.num_steps, args.num_envs) + (1,)).to(device)
    logprobs_backup = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_backup = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_backup = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_backup = torch.zeros((args.num_steps, args.num_envs)).to(device)
    advantages_backup = torch.zeros((args.num_steps, args.num_envs)).to(device)
    returns_backup = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_done_backup = torch.zeros(args.num_envs).to(device)
    next_obs_backup = torch.zeros((args.num_envs,) + envs.single_observation_space.shape).to(device)

    # full replay buffer
    obs_un =  None
    obs_latent = None
    obs_un_train = torch.tensor(envs.envs[0].reset()[0], dtype=torch.float).to(device).unsqueeze(0).repeat(args.classifier_memory,1)
    probs_un_train = None
    # times_full = np.zeros((args.n_capacity,max_steps)+(1,))
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset(seed=args.seed)
    times[0] = torch.tensor(np.array([infos["l"]])).transpose(0,1).to(device)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    video_filenames = set()
    adaptative_success = 0 
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, classifier = classifier)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            times[step] = torch.tensor(np.array([infos["l"]])).transpose(0,1).to(device)
            done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            
        ########################### CLASSIFIER ############################
        # rho_n
        b_batch_obs_rho_n = obs.reshape(-1, *envs.single_observation_space.shape)
        b_batch_times_rho_n = times.reshape(-1, 1)
        mask_obs_rho = (b_batch_times_rho_n >= torch.max(times).item()*args.per_max_step).squeeze(-1)
        b_batch_obs_rho_n = b_batch_obs_rho_n[mask_obs_rho]
        # train the classifier if obs_un_train
        if obs_un_train is not None and obs_un_train.shape[0] >= args.classifier_memory:
            # un_n
            b_batch_obs_un = obs_un_train
            b_batch_probs_un = probs_un_train
            # classifier_epochs
            classifier_epochs = int(b_batch_obs_un.shape[0]/args.classifier_batch_size)*args.classifier_epochs
            for epoch_classifier in range(classifier_epochs):
                # sample rho_n
                idx_step_rho = np.random.randint(0, b_batch_obs_rho_n.shape[0], size = args.classifier_batch_size)
                # sample un_n
                idx_step_un = np.random.randint(0, b_batch_obs_un.shape[0], size = args.classifier_batch_size)
                # mini batch
                mb_rho_n = b_batch_obs_rho_n[idx_step_rho]
                mb_un_n = torch.Tensor(b_batch_obs_un[idx_step_un]).to(device)
                # train the classifier
                classifier_optimizer.zero_grad()
                loss = classifier.ce_loss_ppo(batch_q=mb_rho_n, batch_p=mb_un_n)
                loss.backward()
                classifier_optimizer.step()
                # log the loss
                writer.add_scalar("losses/loss_classifier", loss.item(), global_step)
        
        ############################ REWARD ##################################
        # update the reward
        rewards_nn = classifier(obs).detach().squeeze(-1)
        # normalize the reward
        # rewards = (rewards_nn - rewards_nn.mean()) / (rewards_nn.std() + 1e-1) * args.ratio_reward
        rewards = rewards_nn * args.ratio_reward
        # mask entropy
        mask_entropy = (args.treshold_entropy <= rewards_nn).float()
        # success
        cs.update((((rewards_nn >= args.treshold_entropy).float().mean().item() >= args.per_threshold) & (rewards_nn.mean(dim=0) >= args.treshold_success)).item())
        success = cs.get()
        

        ########################### UPDATE THE BUFFER ############################
        # obs_un
        mask_add =  ~mask_obs_rho
        obs_un = torch.cat([obs_un, obs.reshape(-1, *envs.single_observation_space.shape).clone()], dim=0) if (obs_un is not None) else obs.reshape(-1, *envs.single_observation_space.shape).clone()
        # obs_train & probs_train                                                                                                                       
        obs_un_train = update_train(obs_un, obs_un_train, frac = args.frac_wash, 
                                    capacity = args.classifier_memory, n_past = args.start_explore, 
                                    n_rollouts = args.num_rollouts, max_steps = max_steps, 
                                    n_envs = args.num_envs, classifier = classifier, device = device)  if success  else obs_un_train
        

        ########################### PPO UPDATE ###############################
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
        
        ########################### BACKUP BUFFER ############################
        if update == 1 or classifier(obs_backup).mean().item() < classifier(obs).mean().item():
            print('update backup')
            obs_backup = obs.clone()
            actions_backup = actions.clone()
            times_backup = times.clone()
            logprobs_backup = logprobs.clone()
            rewards_backup = rewards.clone()
            dones_backup = dones.clone()
            values_backup = values.clone()
            advantages_backup = advantages.clone()
            returns_backup = returns.clone()
            mask_entropy_backup = mask_entropy.clone()
            next_obs_backup = next_obs.clone()
            next_done_backup = next_done.clone()
        else : 
            with torch.no_grad():
                rewards_backup = classifier(obs_backup).squeeze(-1) * args.ratio_reward
                next_value_backup = agent.get_value(next_obs_backup).reshape(1, -1)
                advantages_backup = torch.zeros_like(rewards_backup).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done_backup
                        nextvalues = next_value_backup
                    else:
                        nextnonterminal = 1.0 - dones_backup[t + 1]
                        nextvalues = values_backup[t + 1]
                    delta = rewards_backup[t] + args.gamma * nextvalues * nextnonterminal - values_backup[t]
                    advantages_backup[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

        # flatten the batch
        b_obs_train = torch.cat([obs_backup, obs], dim=0).reshape(-1, *envs.single_observation_space.shape)
        b_logprobs_train = torch.cat([logprobs_backup, logprobs], dim=0).reshape(-1)
        b_actions_train = torch.cat([actions_backup, actions], dim=0).reshape(-1, *envs.single_action_space.shape)
        b_advantages_train = torch.cat([advantages_backup, advantages], dim=0).reshape(-1)
        if args.norm_adv:
            b_advantages_train = (b_advantages_train - b_advantages_train.mean()) / (b_advantages_train.std() + 1e-8)
        b_returns_train = torch.cat([returns_backup, returns], dim=0).reshape(-1)
        b_values_train = torch.cat([values_backup, values], dim=0).reshape(-1)
        b_mask_train = torch.cat([mask_entropy_backup, mask_entropy], dim=0).reshape(-1)
        # Optimizing the policy and value network
        b_inds_train = np.arange(2*args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds_train)
            for start in range(0, 2*args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds_train[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs_train[mb_inds], b_actions_train[mb_inds])
                logratio = newlogprob - b_logprobs_train[mb_inds]
                ratio = logratio.exp()
                mask_mb = b_mask_train[mb_inds]
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages_train[mb_inds]
                # if args.norm_adv:
                #     mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss3 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef_mask, 1 + args.clip_coef_mask)
                # pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                pg_loss = (torch.max(pg_loss1, pg_loss2)*(1-mask_mb) + torch.max(pg_loss1, pg_loss3)*mask_mb).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns_train[mb_inds]) ** 2
                    v_clipped = b_values_train[mb_inds] + torch.clamp(
                        newvalue - b_values_train[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns_train[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns_train[mb_inds]) ** 2).mean()

                entropy_loss = (entropy*mask_mb).mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef 
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values_train.cpu().numpy(), b_returns_train.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/advantages_mean", mb_advantages.mean(), global_step)
        writer.add_scalar("charts/rewads_mean", rewards_nn.mean(), global_step)
        writer.add_scalar("charts/rewads_std", rewards_nn.std(), global_step)
        writer.add_scalar("charts/reward_max", rewards_nn.max(), global_step)
        writer.add_scalar("charts/reward_min", rewards_nn.min(), global_step)

        print("SPS:", int(global_step / (time.time() - start_time)))
        print(f"global_step={global_step}")
        print('update : ',update)
        print('reward mean : ',rewards_nn.mean())
        print('reward max : ',rewards.max())
        print('reward min : ',rewards.min())
        print('success : ',success)
        # print('success : ',success)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.track and args.capture_video:
            for filename in os.listdir(f"videos/{run_name}"):
                if filename not in video_filenames and filename.endswith(".mp4"):
                    wandb.log({f"videos": wandb.Video(f"videos/{run_name}/{filename}")})
                    video_filenames.add(filename)
        if update % args.fig_frequency == 0  and global_step > 0:
            if args.make_gif : 
                env_plot.gif(obs_un, obs_un_train, obs.clone(), classifier, device)
            if args.plotly:
                env_plot.plotly(obs_un, obs_un_train, classifier, device)
    envs.close()
    writer.close()
