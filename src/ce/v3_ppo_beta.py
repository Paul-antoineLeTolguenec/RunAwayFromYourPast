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
from torch.utils.tensorboard import SummaryWriter
# from stable_baselines3.common.buffers import ReplayBuffer
from src.ce.classifier import Classifier
from src.ce.vector_encoding import VE
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
    parser.add_argument("--num_rollouts", type=int, default=8,
        help="the number of rollouts ")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=64,
        help="the number of mini-batches")
    parser.add_argument("--minibatch-size", type=int, default=64,
                        help="the size of the mini-batch")
    parser.add_argument("--update-epochs", type=int, default=16,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.05,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    # classifier
    parser.add_argument("--classifier-lr", type=float, default=2e-3)
    parser.add_argument("--classifier-batch-size", type=int, default=128)
    parser.add_argument("--classifier-frequency", type=int, default=1)
    parser.add_argument("--classifier-epochs", type=int, default=32)
    parser.add_argument("--un-n-past", type=int, default=16)
    # n agent
    parser.add_argument("--n-agent", type=int, default=5)
    parser.add_argument("--lamda-im", type=float, default=5.0)
    parser.add_argument("--ratio-reward", type=float, default=1.0)
    args = parser.parse_args()
    args.num_envs = args.n_agent
    # args.num_steps = args.num_steps // args.num_envs
    # fmt: on
    return args


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
        # classifier epoch 
        args.classifier_epochs = (args.num_steps * args.num_envs) // args.classifier_batch_size
    # update batch size and minibatch size
    args.batch_size = int(args.num_envs * args.num_steps)
    # args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_minibatches = int(args.batch_size // args.minibatch_size)
    print('mini batch size',args.minibatch_size)
    print('num minibatches',args.num_minibatches)
    print('batch size',args.batch_size)
    print('classifier epochs',args.classifier_epochs)
    # Agent
    agent = Agent(envs, args.n_agent).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    classifier = Classifier(observation_space=envs.single_observation_space, device=device, env_max_steps = max_steps, learn_z=True, n_agent = args.n_agent, n_past=args.un_n_past)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=args.classifier_lr, eps=1e-5)
    # vector encoding
    ve = VE(n = args.n_agent, device = device, prob = torch.ones(args.n_agent)/args.n_agent)
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
    # times_full = np.zeros((args.n_capacity,max_steps)+(1,))
    # sample n z 
    z = ve.z

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset(seed=args.seed)
    times[0] = torch.tensor(np.array([infos["l"]])).transpose(0,1).to(device)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    video_filenames = set()
    
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

            # if terminated, reset the env

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
            # print('step',step)
            # for info in infos["final_info"]:
            #     # Skip the envs that are not done
            #     if info is None:
            #         continue
            #     print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            #     writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            #     writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
           
       
        # add to buffer
        # reshape (num_rollouts, num_envs, max_steps, obs_shape)
        # obs_rho_n = obs.cpu().numpy().reshape(args.num_rollouts * args.num_envs, max_steps, *envs.single_observation_space.shape)
        # times_rho_n = times.cpu().numpy().reshape(args.num_rollouts * args.num_envs, max_steps, 1)
        # train the classifier
        if update%args.classifier_frequency == 0 and update > 1:
            batch_rho_n = obs.reshape(args.num_rollouts * args.num_envs * max_steps, *envs.single_observation_space.shape)
            batch_rho_n_z = zs.reshape(args.num_rollouts * args.num_envs * max_steps, 1)
            batch_rho_n_times = times.reshape(args.num_rollouts * args.num_envs * max_steps, 1)
            for epoch in range(args.classifier_epochs):
                # sample from rho_n
                idx_step_rho = np.random.randint(0, args.num_rollouts * args.num_envs * max_steps, args.classifier_batch_size)
                batch_rho_n_ext = batch_rho_n[idx_step_rho]
                batch_rho_n_z_ext = batch_rho_n_z[idx_step_rho]
                batch_rho_n_times_ext = batch_rho_n_times[idx_step_rho]
                # sample from un
                # idx_z_un = np.random.randint(0, args.n_agent, args.classifier_batch_size)
                idx_ep_un = np.random.randint(max(0,(update-args.un_n_past)*args.num_rollouts), update*args.num_rollouts, args.classifier_batch_size)
                idx_step_un = np.random.randint(0, max_steps, args.classifier_batch_size)
                batch_un = torch.Tensor(obs_un[:, idx_ep_un, idx_step_un]).to(device)
                # batch_un_ext = batch_un.permute(1,0,2)
                # train the classifier
                classifier_optimizer.zero_grad()
                classifier_loss = classifier.ce_loss_ppo(batch_rho_n_ext, batch_rho_n_times_ext, batch_un, batch_rho_n_z_ext)
                classifier_loss.backward()
                classifier_optimizer.step()
                writer.add_scalar("losses/classifier_loss", classifier_loss.item(), global_step)

        # update reward
        with torch.no_grad():
            log_p_rho_un = classifier(obs).detach().squeeze(-1)
            # compute p(z|s)
            p_s_z = torch.gather(torch.softmax(classifier.forward_z(obs),dim=-1), -1, (zs-1).type(torch.int64)).squeeze(-1)
            log_p_s_z = torch.log(p_s_z + 1e-8)
            p = torch.exp(log_p_rho_un)
            # normalize 
            # im = (im - torch.mean(im))/(torch.std(im) + 1e-8)
            # log_p_rho_un = (log_p_rho_un - torch.mean(log_p_rho_un))/(torch.std(log_p_rho_un) + 1e-8)
            # rewards 
            # rewards = log_p_rho_un + args.lamda_im*im
            rewards = log_p_rho_un + args.lamda_im*log_p_s_z
            print('max log_p_s_z',torch.max(log_p_s_z))
            print('min log_p_s_z',torch.min(log_p_s_z))
            print('mean log_p_s_z',torch.mean(log_p_s_z))
            print('max log_p_rho_un',torch.max(log_p_rho_un))
            print('min log_p_rho_un',torch.min(log_p_rho_un))
            print('mean log_p_rho_un',torch.mean(log_p_rho_un))
            print('std log_p_rho_un',torch.std(log_p_rho_un))
        # add to buffer
        
        obs_un[:, args.num_rollouts * (update-1):args.num_rollouts * update] = obs.permute(1,0,2).reshape(args.num_envs, args.num_rollouts, max_steps, *envs.single_observation_space.shape).cpu().numpy()
        z_un[:, args.num_rollouts * (update-1):args.num_rollouts * update] = zs.permute(1,0,2).reshape(args.num_envs, args.num_rollouts, max_steps, 1).cpu().numpy()


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
        print(f"global_step={global_step}")
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.track and args.capture_video:
            for filename in os.listdir(f"videos/{run_name}"):
                if filename not in video_filenames and filename.endswith(".mp4"):
                    wandb.log({f"videos": wandb.Video(f"videos/{run_name}/{filename}")})
                    video_filenames.add(filename)
        if update % args.fig_frequency == 0 and args.make_gif and global_step > 0:
            # clear the plot
            # env_plot.ax.clear()
            # reset the limits
            # env_plot.reset_lim_fig()
            # Plotting measure 
            # m_n = classifier(torch.Tensor(obs_un[:, args.num_rollouts * (update-1) :args.num_rollouts * update]).to(device), torch.Tensor(z_un[:, args.num_rollouts * (update-1) :args.num_rollouts * update]).to(device)).detach().cpu().numpy()
            # print('m_n',m_n.shape)
            # Plotting the environment
            data_to_plot  = obs_un[:, args.num_rollouts * (update-1) :args.num_rollouts * update].reshape(args.num_envs, args.num_rollouts* max_steps, *envs.single_observation_space.shape)
            for z_t in range(args.n_agent):
                env_plot.ax.scatter(data_to_plot[z_t,:,0], data_to_plot[z_t,:,1], label = f'z = {z_t}', c = colors[z_t], s=1)

            # save fig env_plot
            env_plot.figure.canvas.draw()
            image = np.frombuffer(env_plot.figure.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(env_plot.figure.canvas.get_width_height()[::-1] + (3,))
            writer_gif.append_data(image)
            # iter_plot
            iter_plot += 1
    envs.close()
    writer.close()
