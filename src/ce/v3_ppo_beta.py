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
from src.ce.vector_encoding import VE
from src.ce.discounted_ucb import DiscountedUCB
from envs.continuous_maze import Maze
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
    parser.add_argument("--env-type", type=str, default="Maze")
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
    parser.add_argument("--fig_frequency", type=int, default=1)
    parser.add_argument("--make-gif", type=bool, default=True)
    parser.add_argument("--episodic-return", type=bool, default=True)

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Ur",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=int(1e7),
        help="total timesteps of the experiments")
    parser.add_argument("--n-capacity", type=int, default=10**5,
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
    parser.add_argument("--num-minibatches", type=int, default=64,
        help="the number of mini-batches")
    parser.add_argument("--minibatch-size", type=int, default=128,
                        help="the size of the mini-batch")
    parser.add_argument("--update-epochs", type=int, default=32,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.05,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=1.0,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    # classifier
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-batch-size", type=int, default=256)
    parser.add_argument("--classifier-memory", type=int, default=2000)
    parser.add_argument("--classifier-frequency", type=int, default=1)
    parser.add_argument("--classifier-epochs", type=int, default=1)
    parser.add_argument("--frac-wash", type=float, default=1/4, help="fraction of the buffer to wash")
    parser.add_argument("--boring-n", type=int, default=4)
    parser.add_argument("--treshold-entropy", type=float, default=0.0)
    parser.add_argument("--ratio-speed", type=float, default=0.5)
    parser.add_argument("--tau-exp-rho", type=float, default=0.25)
    # n agent
    parser.add_argument("--n-agent", type=int, default=5)
    parser.add_argument("--lamda-im", type=float, default=1.0)
    parser.add_argument("--ratio-reward", type=float, default=1.0)
    parser.add_argument("--learning-explore-start", type=int, default=8)
    args = parser.parse_args()
    args.num_envs = args.n_agent
    args.classifier_memory*= args.n_agent
    # args.num_steps = args.num_steps // args.num_envs
    # fmt: on
    return args


def wash(classifier, obs_train, prob_obs_train, obs_un_n, 
        prob_obs_un_n, num_rollouts, max_steps, 
        num_envs, boring_n, update_n_agent,
        classifier_memory, frac_wash, ratio_speed, device):
    """ Wash the buffer """
    # last rho_n 
    last_obs_rho_n = np.concatenate([obs_un_n[i, : (update_n_agent[i]-boring_n[i])*num_rollouts][-num_rollouts : ].reshape(-1, obs_un_n.shape[-1]) for i in range(num_envs)],axis=0)
    # portion to remove
    size_to_remove = int(frac_wash*obs_train.shape[0])
    # big batch un 
    size_per_agent = (classifier_memory//num_envs)
    idx_ep_un_n = np.zeros((num_envs, size_per_agent),dtype=int)
    idx_step_un_n = np.zeros((num_envs, size_per_agent),dtype=int)
    idx_z_un_n = np.concatenate([np.ones((size_per_agent,1))*i for i in range(num_envs)],axis=0).astype(int)
    for i in range(num_envs):
        idx_ep_un_n[i] = np.random.randint(0,(update_n_agent[i]-boring_n[i])*num_rollouts, size = size_per_agent)
        idx_step_un_n[i] = np.random.randint(0, max_steps, size = size_per_agent)
    big_batch_un_n = np.concatenate([obs_un_n[i, : (update_n_agent[i]-boring_n[i])*num_rollouts][idx_ep_un_n[i], idx_step_un_n[i]] for i in range(num_envs)],axis=0)
    big_batch_un_n_prob = np.concatenate([prob_obs_un_n[i, : (update_n_agent[i]-boring_n[i])*num_rollouts][idx_ep_un_n[i], idx_step_un_n[i]] for i in range(num_envs)],axis=0)
    idx_ep_un_n = idx_ep_un_n.reshape(-1,1)
    idx_step_un_n = idx_step_un_n.reshape(-1,1)
    # update prob_obs_train
    prob_obs_train = (1-torch.sigmoid(classifier(torch.Tensor(obs_train).to(device)))).detach().cpu().numpy().squeeze(-1)
    # normalize prob_obs_train
    prob_obs_train = prob_obs_train/prob_obs_train.sum()
    # update prob_obs_un_n
    prob_obs_un_n_norm = (torch.sigmoid(classifier(torch.Tensor(big_batch_un_n).to(device)))).detach().cpu().numpy()
    # update prob_obs_un_n
    prob_obs_un_n[ idx_z_un_n[:,0], idx_ep_un_n[:,0], idx_step_un_n[:,0]] = prob_obs_un_n_norm
    # mask
    mask_delta_old_new = (0 < (prob_obs_un_n_norm.reshape(-1,1)-big_batch_un_n_prob))
    prob_obs_un_n_norm = prob_obs_un_n_norm + mask_delta_old_new*ratio_speed
    # normalize prob_obs_un_n
    prob_obs_un_norm = prob_obs_un_n_norm/prob_obs_un_n_norm.sum()
    # choose the index to remove from obs_train
    idx_to_remove = np.random.choice(obs_train.shape[0], p=prob_obs_train, size = size_to_remove + last_obs_rho_n.shape[0], replace = True)
    # idx_to_remove = np.random.choice(obs_train.shape[0], p=prob_obs_train, size = size_to_remove, replace = True)
    # choose the index to add from obs_un_n
    idx_to_add = np.random.choice(big_batch_un_n.shape[0], p=prob_obs_un_norm.reshape(-1), size = size_to_remove, replace = True)
    # update the buffer
    obs_train[idx_to_remove[:size_to_remove]] = big_batch_un_n[idx_to_add]
    obs_train[idx_to_remove[size_to_remove:]] = last_obs_rho_n

    # update prob_obs_train
    prob_obs_train = (torch.sigmoid(classifier(torch.Tensor(obs_train).to(device)))).detach().cpu().numpy().squeeze(-1)
    # # normalize prob_obs_train
    prob_obs_train = prob_obs_train/prob_obs_train.sum()
    return obs_train, prob_obs_train, prob_obs_un_n

def make_env(env_id, idx, capture_video, run_name, gamma, env_type = "gym"):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id) if env_type == "gym" else Maze(name=env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, n_agent):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod()+1, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod()+1, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x, z):
        return self.critic(torch.cat((x,z),-1))

    def get_action_and_value(self, x, z, action=None):
        x = torch.cat((x,z),-1)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
        # env to plot 
        env_plot = Maze(name = args.env_id, fig = True)
        # iter_plot 
        iter_plot = 0
        if not os.path.exists('gif'):
            os.makedirs('gif')
        writer_gif = imageio.get_writer('gif/v2_ppo_beta.mp4', fps=2)
        # generate n_agent different colors for matplotlib
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma, args.env_type) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    if args.episodic_return:
        max_steps = envs.envs[0].max_steps
        args.num_steps = max_steps * args.num_rollouts
    # update batch size and minibatch size
    args.batch_size = int(args.num_envs * args.num_steps)
    # args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_minibatches = int(args.batch_size // args.minibatch_size)
    print('mini batch size',args.minibatch_size)
    print('num minibatches',args.num_minibatches)
    print('batch size',args.batch_size)
    # Agent
    agent = Agent(envs, args.n_agent).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    classifier = Classifier(observation_space=envs.single_observation_space, device=device, env_max_steps = max_steps, learn_z=True, n_agent = args.n_agent)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=args.classifier_lr, eps=1e-5)
    # vector encoding
    ve = VE(n = args.n_agent, device = device, prob = torch.ones(args.n_agent)/args.n_agent)
    # discounted ucb
    ducb = DiscountedUCB(n_arms = args.n_agent, gamma= 0.99)
    # fixed weights reward
    weights_reward = [ i/args.n_agent for i in range(args.n_agent)]
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    zs = torch.zeros((args.num_steps, args.num_envs) + (1,)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    times = torch.zeros((args.num_steps, args.num_envs) + (1,)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # full replay buffer
    obs_un = np.zeros((args.num_envs,args.n_capacity,max_steps) + envs.single_observation_space.shape)
    z_un = np.zeros((args.num_envs, args.n_capacity, max_steps) + (1,))
    probs_un =  np.ones((args.num_envs, args.n_capacity, max_steps) + (1,))
    obs_un_train = np.zeros((args.classifier_memory,envs.single_observation_space.shape[0]))
    probs_un_train = np.zeros((args.classifier_memory,1))
    # times_full = np.zeros((args.n_capacity,max_steps)+(1,))
    # sample n z 
    z = ve.z
    boring_n_agent = np.ones(args.n_agent,dtype=int)*args.boring_n
    update_n_agent = np.zeros(args.n_agent,dtype=int)
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset(seed=args.seed)
    times[0] = torch.tensor(np.array([infos["l"]])).transpose(0,1).to(device)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    video_filenames = set()
    # UCB Choice 
    for env in range(args.num_envs):
        # select the arm
        idx_arm = ducb.select_arm()
        # update the arm with 0 reward
        ducb.update_arm(idx_arm, 0)
        # update z 
        z[env] = idx_arm + 1
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
            zs[step] = z.unsqueeze(-1)
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, z.unsqueeze(-1))
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            times[step] = torch.tensor(np.array([infos["l"]])).transpose(0,1).to(device)
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)*args.ratio_reward
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue
            if True in done:
                # reset z 
                z = ve.z
    
        ########################### CLASSIFIER ############################
        
        # rho_n
        b_batch_obs_rho_n = obs.permute(1,0,2).reshape(args.num_envs*args.num_rollouts, max_steps, *envs.single_observation_space.shape)
        b_batch_times_rho_n = times.permute(1,0,2).reshape(args.num_envs*args.num_rollouts, max_steps, 1).cpu().numpy()
        b_batch_z_rho_n = zs.permute(1,0,2).reshape(args.num_envs*args.num_rollouts, max_steps, 1)
        # train the classifier
        if update%args.classifier_frequency == 0 and update > args.boring_n + int(args.classifier_memory/(args.num_rollouts*args.num_envs*max_steps)):
            # un_n
            b_batch_obs_un = obs_un_train
            b_batch_probs_un = probs_un_train
            ratio_classifier = args.classifier_memory/(args.num_rollouts*args.num_envs*max_steps)
            # args.classifier_epochs
            args.classifier_epochs = int(b_batch_obs_un.shape[0]/args.classifier_batch_size)*2
            for epoch_classifier in range(args.classifier_epochs):
                # sample rho_n
                idx_ep_rho = np.random.randint(0, args.num_rollouts*args.num_envs, size = args.classifier_batch_size)
                idx_step_rho = (exp_dec(args.classifier_batch_size,tau = args.tau_exp_rho)*max_steps).astype(int)
                # sample un_n
                idx_step_un = np.random.choice(b_batch_obs_un.shape[0], p=b_batch_probs_un, size = args.classifier_batch_size, replace=True)
                # idx_step_un = np.random.randint(0, b_batch_obs_un.shape[0], size = args.classifier_batch_size)
                # mini batch
                mb_rho_n = b_batch_obs_rho_n[idx_ep_rho, idx_step_rho]
                mb_rho_n_times = b_batch_times_rho_n[idx_ep_rho, idx_step_rho]
                mb_rho_n_z = b_batch_z_rho_n[idx_ep_rho, idx_step_rho]
                mb_un_n = torch.Tensor(b_batch_obs_un[idx_step_un]).to(device)
                # train the classifier
                classifier_optimizer.zero_grad()
                loss = classifier.ce_loss_ppo(batch_q=mb_rho_n, batch_p=mb_un_n, batch_q_z=mb_rho_n_z)
                loss.backward()
                classifier_optimizer.step()
                # log the loss
                writer.add_scalar("losses/loss_classifier", loss.item(), global_step)
        
        ############################ REWARD ##################################
        with torch.no_grad():
            # log(rho_n/un_n)
            log_p_rho_un_nn = classifier(obs).detach().squeeze(-1)
            # normalize on dim 0
            log_p_rho_un = (log_p_rho_un_nn - torch.mean(log_p_rho_un_nn, dim=0).unsqueeze(0))/(torch.std(log_p_rho_un_nn, dim=0).unsqueeze(0) + 1e-8)

            # p(z|s)
            p_s_z = torch.gather(torch.softmax(classifier.forward_z(obs),dim=-1), -1, (zs-1).type(torch.int64)).squeeze(-1)
            log_p_s_z_nn = torch.log(p_s_z + 1e-8)
            # normalize on dim 0
            log_p_s_z = (log_p_s_z_nn - torch.mean(log_p_s_z_nn, dim=0).unsqueeze(0))/(torch.std(log_p_s_z_nn, dim=0).unsqueeze(0) + 1e-8)

            # rewards
            rewards = log_p_s_z if update < args.learning_explore_start else log_p_rho_un + args.lamda_im*log_p_s_z
            # mask rewards_nn
            mask_rewards = (0 < log_p_rho_un_nn).float()
            # mask boring_n
            mask_boring_n = (mask_rewards.sum(dim=0) < args.num_rollouts*max_steps/2).int()
            # update boring_n
            boring_n_agent = np.minimum(np.maximum((boring_n_agent+2*mask_boring_n.cpu().numpy()-1),np.ones(args.n_agent)*args.boring_n),np.ones(args.n_agent)*args.boring_n*4).astype(int)
            # mask entropy
            mask_entropy = (args.treshold_entropy <= log_p_rho_un_nn).float()
        ########################### UPDATE UCB ############################
            # reward per arm
            reward_per_arm = torch.mean(log_p_rho_un_nn, dim = 0)
            # rank the arm
            rank_arm = torch.argsort(reward_per_arm, descending = True)
        ########################### UPDATE THE BUFFER ############################
        # update the buffer for each agent
        for i in range(args.n_agent):
            obs_un[:, args.num_rollouts * (update_n_agent[i]-1):args.num_rollouts * update_n_agent[i]] = obs.permute(1,0,2).reshape(args.num_envs, args.num_rollouts, max_steps, *envs.single_observation_space.shape).cpu().numpy()
            z_un[:, args.num_rollouts * (update_n_agent[i]-1):args.num_rollouts * update_n_agent[i]] = zs.permute(1,0,2).reshape(args.num_envs, args.num_rollouts, max_steps, 1).cpu().numpy()
        
        # update un_train
        obs_un_train, probs_un_train, probs_un = wash(classifier, 
        obs_un_train, probs_un_train, 
        obs_un, probs_un, 
        args.num_rollouts, max_steps, 
        args.num_envs, boring_n_agent, update_n_agent,
        args.classifier_memory, args.frac_wash, 
        args.ratio_speed, device) if (update > args.boring_n) else (obs_un[:, :(args.classifier_memory//(max_steps*args.num_envs))].reshape(-1, envs.single_observation_space.shape[0]),probs_un_train,probs_un)
        ########################### PPO UPDATE ###############################
       
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs,z.unsqueeze(-1)).reshape(1, -1)
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
        b_zs = zs.reshape((-1,) + (1,))
        b_mask = mask_entropy.reshape(-1)
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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(x = b_obs[mb_inds], z = b_zs[mb_inds], action = b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mask_mb = b_mask[mb_inds]
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

                entropy_loss = (entropy*mask_mb).sum()/(mask_mb.sum()+1)
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
        print('max log_p_s_z',torch.max(log_p_s_z_nn))
        print('min log_p_s_z',torch.min(log_p_s_z_nn))
        print('mean log_p_s_z',torch.mean(log_p_s_z_nn))
        print('max log_p_rho_un',torch.max(log_p_rho_un_nn))
        print('min log_p_rho_un',torch.min(log_p_rho_un_nn))
        print('mean log_p_rho_un',torch.mean(log_p_rho_un_nn))
        print('std log_p_rho_un',torch.std(log_p_rho_un_nn))
        print('sum mask_rewards dim 0',mask_rewards.sum(dim=0))
        print('boring_n',boring_n_agent)
        print("SPS:", int(global_step / (time.time() - start_time)))
        print(f"global_step={global_step}")
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.track and args.capture_video:
            for filename in os.listdir(f"videos/{run_name}"):
                if filename not in video_filenames and filename.endswith(".mp4"):
                    wandb.log({f"videos": wandb.Video(f"videos/{run_name}/{filename}")})
                    video_filenames.add(filename)
        if update % args.fig_frequency == 0 and args.make_gif and global_step > 0:
            with torch.no_grad():
                # clear the plot
                env_plot.ax.clear()
                # reset the limits
                env_plot.reset_lim_fig()

                # plot measure 
                data_to_plot  = torch.Tensor(obs_un[:, :args.num_rollouts * update].reshape(args.num_envs*args.num_rollouts*update* max_steps, *envs.single_observation_space.shape)).to(device)
                # # Plotting measure 
                m_n = classifier(data_to_plot).detach().cpu().numpy().flatten()
                # # Plotting the environment
                env_plot.ax.scatter(data_to_plot[:,0], data_to_plot[:,1], c = m_n, s=1)
                # color_treshold = 'r'
                # # mask
                # mask = (m_n <= -5)
                # # arg mask
                # arg_mask = np.argwhere(mask)
                # # scatter red dot if m_n <= -5
                # env_plot.ax.scatter(data_to_plot[arg_mask,0], data_to_plot[arg_mask,1], s=1, c = color_treshold)
                # plot obs_un_train 
                env_plot.ax.scatter(obs_un_train[:,0], obs_un_train[:,1], s=1, c = 'b')

                # plot per skill
                # data_to_plot  = obs_un[:, args.num_rollouts * (update-1) :args.num_rollouts * update].reshape(args.num_envs, args.num_rollouts* max_steps, *envs.single_observation_space.shape)
                # print('data_to_plot',data_to_plot.shape)
                # for z_t in range(args.n_agent):
                    # env_plot.ax.scatter(data_to_plot[z_t,:,0], data_to_plot[z_t,:,1], label = f'z = {z_t}', c = colors[z_t], s=1)

                # save fig env_plot
                env_plot.figure.canvas.draw()
                image = np.frombuffer(env_plot.figure.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(env_plot.figure.canvas.get_width_height()[::-1] + (3,))
                writer_gif.append_data(image)
                # iter_plot
                iter_plot += 1
    envs.close()
    writer.close()
