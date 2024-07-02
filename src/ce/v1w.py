# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/rpo/#rpo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import gym as gym_old
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
# import specific 
from src.ce.classifier import Classifier
from src.utils.replay_buffer import ReplayBuffer
from src.utils.wandb_utils import send_video, send_matrix, send_dataset


from envs.wenv import Wenv
from envs.config_env import config



@dataclass
class Args:
    # XP RECORD
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
    wandb_project_name: str = "contrastive_test"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_data: bool = True
    """whether to save the data of the experiment"""

    # GIF
    make_gif: bool = True
    """if toggled, will make gif """
    plotly: bool = False
    """if toggled, will use plotly instead of matplotlib"""
    fig_frequency: int = 1
    """the frequency of logging the figures"""
    shannon_compute_freq: int = 5
    """the frequency of computing shannon entropy"""

    # RPO SPECIFIC
    env_id: str = "HalfCheetah-v3"
    """the id of the environment"""
    total_timesteps: int = 5_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
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
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_mask_coef: float = 0.2
    """the mask clipping coefficient"""
    clip_vloss: bool = False #True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    ent_mask_coef: float = 0.1
    """coefficient of the entropy mask"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # CLASSIFIER SPECIFIC
    classifier_lr: float = 1e-4
    """the learning rate of the classifier"""
    classifier_epochs: int = 1
    """the number of epochs to train the classifier"""
    classifier_batch_size: int = 256
    """the batch size of the classifier"""
    feature_extractor: bool = False
    """if toggled, a feature extractor will be used"""
    percentage_time: float = 0/4
    """the percentage of the time to use the classifier"""
    epsilon: float = 1e-3
    """the epsilon of the classifier"""
    lambda_init: float = 10.0 #50 in mazes
    """the lambda of the classifier"""
    bound_spectral: float = 1.0
    """the bound spectral of the classifier"""
    clip_lim: float = 100.0
    """the clipping limit of the classifier"""
    adaptive_sampling: bool = False
    """if toggled, the sampling will be adaptive"""
    lip_cte: float = 1.0
    """the lip constant"""
    use_sigmoid: bool = True
    """if toggled, the sigmoid will be used"""

    

    # RHO SPECIFIC
    episodic_return: bool = True
    """if toggled, the episodic return will be used"""
    polyak: float = 0.75
    """the polyak averaging coefficient"""
    n_rollouts: int = 1
    """the number of rollouts"""
    keep_extrinsic_reward: bool = False
    """if toggled, the extrinsic reward will be kept"""
    start_explore: int = 1
    """the number of updates to start exploring"""
    coef_intrinsic : float = 1.0
    """the coefficient of the intrinsic reward"""
    coef_extrinsic : float = 1.0
    """the coefficient of the extrinsic reward"""
    beta_ratio: float = 1/256
    """the ratio of the beta"""
    beta_min: float = 1/1024
    """the minimum of the beta"""
    beta_max: float = 1/128
    """the maximum of the beta"""
    adaptive_beta: bool = False
    """if toggled, the beta will be adaptive"""
    beta_increment_bool: bool = False
    """if toggled, the beta will be incremented"""

    nb_max_steps: int = 20_000
    """the maximum number of step in un"""
    gamma_cte: float = 0.0
    """the constant gamma"""

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
        env = gym_old.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym_old.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env = gym.wrappers.ClipAction(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, rpo_alpha = 0.0):
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
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

def update_probs(obs_un, classifier, device):
    with torch.no_grad():
        # probs batch un
        batch_probs_un = (torch.sigmoid(classifier(torch.Tensor(obs_un).to(device)))).detach().cpu().numpy().squeeze(-1)
        batch_probs_un_norm = batch_probs_un/batch_probs_un.sum()
    return batch_probs_un_norm

def update_un(obs_un, next_obs_un, actions_un, rewards_un,  dones_un, times_un,
              obs_reshaped, next_obs_reshaped, actions_reshaped, rewards_reshaped, dones_reshaped, times_reshaped,
              args):
    n_batch = int(obs_un.shape[0]*args.beta_ratio)
    idx_un = np.random.randint(0, obs_un.shape[0], size = n_batch)
    idx_rho = np.random.randint(0, obs_reshaped.shape[0], size = n_batch)
    obs_un[idx_un] = obs_reshaped[idx_rho].copy()
    next_obs_un[idx_un] = next_obs_reshaped[idx_rho].copy()
    actions_un[idx_un] = actions_reshaped[idx_rho].copy()
    rewards_un[idx_un] = rewards_reshaped[idx_rho].copy()
    dones_un[idx_un] = dones_reshaped[idx_rho].copy()
    times_un[idx_un] = times_reshaped[idx_rho].copy()
    return obs_un, next_obs_un, actions_un, rewards_un, dones_un, times_un

if __name__ == "__main__":
    from src.utils.argparse_test import parse_args
    args = parse_args(Args)
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
    # coverage check env 
    env_check = Wenv(env_id=args.env_id,
                    render_bool_matplot=False,
                    xp_id=run_name,
                    **config[args.env_id])
    # MAX STEPS
    max_steps = config[args.env_id]['kwargs']['max_episode_steps']
    args.num_steps = max_steps * args.n_rollouts +1
    # BATCH CALCULATION
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    # args.classifier_epochs = (args.num_steps*args.num_envs // args.classifier_batch_size) * args.classifier_epochs
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
    # AGENT
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # CLASSIFIER
    classifier = Classifier(envs.single_observation_space, 
                            env_max_steps=max_steps,
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
    replay_buffer = ReplayBuffer(capacity= int(1e6),
                                observation_space= envs.single_observation_space,
                                action_space= envs.single_action_space,
                                device=device)
    # RPO: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs) + (1,)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs) + (1,)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs) + (1,)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)+ (1,)).to(device)
    times = torch.zeros((args.num_steps, args.num_envs)+ (1,)).to(device)

    # UN
    # obs_un = torch.tensor(envs.envs[0].reset()[0], dtype=torch.float).unsqueeze(0).repeat(args.num_steps*args.num_envs, 1).cpu().numpy()
    # dones_un = torch.zeros((args.num_steps*args.num_envs, 1)).cpu().numpy()
    obs_un = None
    next_obs_un = None
    actions_un = None
    rewards_un = None
    dones_un = None
    times_un = None

    # INIT Wasserstein Rho Un
    w_rho_un = 0
    max_w_rho_un = 0
    min_w_rho_un = 0
    last_w_rho_un = 0
    beta_increment = 0
    beta_adaptive = args.beta_ratio

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    times[0] = torch.tensor(np.array([infos["l"]])).transpose(0,1).to(device)

    for update in range(1, num_updates + 1):
        if args.episodic_return:
            next_obs, infos = envs.reset(seed=args.seed)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.zeros(args.num_envs).to(device)
            num_updates = args.total_timesteps // args.batch_size
            times[0] = torch.tensor(np.array([infos["l"]])).transpose(0,1).to(device)
        
        # PLAYING IN ENV
        for step in range(0, args.num_steps):
            # coverage assessment 
            env_check.update_coverage(next_obs)
            # ppo
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done.unsqueeze(-1)

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, action = None)
                # values[step] = value.flatten()
                values[step] = value

            actions[step] = action
            logprobs[step] = logprob.unsqueeze(-1)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            times[step] = torch.tensor(np.array([infos["l"]])).transpose(0,1).to(device)
            done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).unsqueeze(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        wandb.log({"specific/episodic_return": info["episode"]["r"], "specific/episodic_length": info["episode"]["l"], "global_step": global_step})
       

       ########################### CLASSIFIER UPDATE ########################################
        total_classification_loss = 0
        total_lipshitz_regu = 0
        if update > args.start_explore:
            # CLASSIFIER TRAINING
            batch_obs_rho = obs.permute(1,0,2).reshape(-1, obs.shape[-1]).to(device)
            batch_dones_rho = dones.permute(1,0,2).reshape(-1).to(device)
            batch_times_rho = times.permute(1,0,2).reshape(-1).to(device)
            prob_unorm = torch.clamp(1/torch.tensor(args.gamma+0.009).pow(batch_times_rho.cpu()),1_00.0)
            prob = prob_unorm[:-1]/prob_unorm[:-1].sum()
            # classifier epoch 
            classifier_epochs = max((obs_un.shape[0] // args.classifier_batch_size) * args.classifier_epochs, (batch_obs_rho.shape[0] // args.classifier_batch_size) * args.classifier_epochs)
            print('classifier_epochs:', classifier_epochs)
            for epoch in range(classifier_epochs):
                # mb rho
                mb_rho_idx = np.random.choice(np.arange(batch_obs_rho.shape[0]-1), args.classifier_batch_size, p=prob.numpy())
                # mb_rho_idx = np.random.randint(0, batch_obs_rho.shape[0]-1, args.classifier_batch_size)
                mb_rho = batch_obs_rho[mb_rho_idx].to(device)
                next_mb_rho = batch_obs_rho[mb_rho_idx+1].to(device)
                done_mb_rho = batch_dones_rho[mb_rho_idx].to(device)
                # mb un
                beta_mb_rho_idx = np.random.randint(0, batch_obs_rho.shape[0]-1, int(args.classifier_batch_size*beta_adaptive))
                beta_mb_un_idx = np.random.randint(0, obs_un.shape[0] - beta_increment ,int(args.classifier_batch_size*(1-beta_adaptive)))
                mb_un = torch.tensor(np.concatenate((obs_un[beta_mb_un_idx], batch_obs_rho[beta_mb_rho_idx].cpu().numpy()), axis=0)).to(device)
                next_mb_un = torch.tensor(np.concatenate((next_obs_un[beta_mb_un_idx], batch_obs_rho[beta_mb_rho_idx+1].cpu().numpy()), axis=0)).to(device)
                done_mb_un = torch.tensor(np.concatenate((dones_un[beta_mb_un_idx], batch_dones_rho[beta_mb_rho_idx].cpu().numpy()), axis=0)).to(device)
                # classifier loss + lipshitz regularization
                loss, _ = classifier.lipshitz_loss_ppo(batch_q= mb_rho, batch_p = mb_un, 
                                                        q_batch_s =  mb_rho, q_batch_next_s = next_mb_rho, q_dones = done_mb_rho,
                                                        p_batch_s = mb_un, p_batch_next_s = next_mb_un, p_dones = done_mb_un,) 
                                                        # N_rho = batch_obs_rho.shape[0], N_un = obs_un.shape[0])       
                classifier_optimizer.zero_grad()
                loss.backward()
                classifier_optimizer.step()
                total_classification_loss += loss.item()/classifier_epochs
                # lambda loss
                _, lipshitz_regu = classifier.lipshitz_loss_ppo(batch_q= mb_rho, batch_p = mb_un, 
                                                        q_batch_s =  mb_rho, q_batch_next_s = next_mb_rho, q_dones = done_mb_rho,
                                                        p_batch_s = mb_un, p_batch_next_s = next_mb_un, p_dones = done_mb_un,)
                                                        # N_rho = batch_obs_rho.shape[0], N_un = obs_un.shape[0])    
                lambda_loss = classifier.lambda_lip*lipshitz_regu
                classifier_optimizer.zero_grad()
                lambda_loss.backward()
                classifier_optimizer.step()
                total_lipshitz_regu += lipshitz_regu.item()/classifier_epochs



        ############################ INTRINSIC REWARD ########################################
        with torch.no_grad():
            intrinsic_reward = classifier(obs)
            intrinsic_reward += intrinsic_reward.min()
            # obs_d = classifier(obs[:-1])
            # next_obs_d = classifier(obs[1:])
            # intrinsic_reward = torch.cat([(next_obs_d-obs_d), torch.zeros((1, args.num_envs,1)).to(device)], dim=0) * 10.0

        extrinsic_rewards = rewards.clone()
        rewards = args.coef_extrinsic * rewards + args.coef_intrinsic * intrinsic_reward if args.keep_extrinsic_reward else args.coef_intrinsic * intrinsic_reward
        mask_pos = (intrinsic_reward > 0).float()

        ############################ WASSERSTEIN SPEED ESTIMATION ########################################

        w_rho_un = intrinsic_reward.mean().item()
        rate_w_rho = w_rho_un - last_w_rho_un
        last_w_rho_un = w_rho_un
        # BETA INCREMENT
        if args.beta_increment_bool:
            beta_increment += int(args.num_envs * args.num_steps*beta_adaptive) if rate_w_rho < 0 else 0
            beta_increment = int(max(min(beta_increment, obs_un.shape[0]-args.num_envs*args.num_steps*beta_adaptive),0)) if obs_un is not None else 0
        # BETA ADAPTIVE
        max_w_rho_un = max(max_w_rho_un, w_rho_un)
        min_w_rho_un = min(min_w_rho_un, w_rho_un)
        normalized_w_rho = (w_rho_un - min_w_rho_un) / (max_w_rho_un - min_w_rho_un)
        beta_adaptive = args.beta_min + normalized_w_rho * (args.beta_max - args.beta_min) if args.adaptive_beta else args.beta_ratio
        print(f"BETA ADAPTIVE : {beta_adaptive}, BETA INCREMENT : {beta_increment}, W_RHO : {w_rho_un}, RATE : {rate_w_rho}, OBS_UN : {obs_un.shape[0] if obs_un is not None else 0}")

        ########################### UPDATE UN ########################################
        # permute
        obs_permute = obs.permute(1,0,2)
        times_permute = times.permute(1,0,2)
        actions_permute = actions.permute(1,0,2)
        rewards_permute = extrinsic_rewards.permute(1,0,2)
        dones_permute = dones.permute(1,0,2)
        # reshape
        obs_reshaped = obs.reshape(-1, obs_permute.shape[-1]).cpu().numpy()
        actions_reshaped = actions_permute.reshape(-1, actions.shape[-1]).cpu().numpy()
        rewards_reshaped = rewards_permute.reshape(-1).cpu().numpy()
        dones_reshaped = dones_permute.reshape(-1).cpu().numpy()
        times_reshaped = times_permute.reshape(-1).cpu().numpy()
        # update un
        idx_un = np.random.randint(0, obs_reshaped.shape[0]-1, int(beta_adaptive*obs_reshaped.shape[0]))
        if obs_un is None:
                obs_un = obs_reshaped[idx_un]
                next_obs_un = obs_reshaped[idx_un+1]
                actions_un = actions_reshaped[idx_un]
                rewards_un = rewards_reshaped[idx_un]
                dones_un = dones_reshaped[idx_un]
                times_un = times_reshaped[idx_un]

        elif obs_un.shape[0] >= args.nb_max_steps:
            obs_un, next_obs_un, actions_un, rewards_un, dones_un, times_un = update_un(obs_un, next_obs_un, actions_un, rewards_un, dones_un, times_un,
                                                    obs_reshaped[:-1], obs_reshaped[1:], actions_reshaped[:-1], rewards_reshaped[:-1], dones_reshaped[:-1], times_reshaped[:-1], 
                                                    args)
            
        else:
            obs_un = np.concatenate([obs_un, obs_reshaped[idx_un]])
            next_obs_un = np.concatenate([next_obs_un, obs_reshaped[idx_un+1]]) 
            actions_un = np.concatenate([actions_un, actions_reshaped[idx_un]])
            rewards_un = np.concatenate([rewards_un, rewards_reshaped[idx_un]])
            dones_un = np.concatenate([dones_un, dones_reshaped[idx_un]])
            times_un = np.concatenate([times_un, times_reshaped[idx_un]])   

        ########################### PPO UPDATE ########################################

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(obs.shape[0])):
                if t == obs.shape[0] - 1:
                    nextnonterminal = 1.0 - next_done.unsqueeze(-1)
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
        if args.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_mask_pos = mask_pos.reshape(-1)
        # Optimizing the policy and value network
        b_inds = np.arange(b_obs.shape[0])
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, obs.shape[0], args.minibatch_size):
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
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss3 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_mask_coef, 1 + args.clip_mask_coef)
                pg_loss = (torch.max(pg_loss1, pg_loss2)*(1-b_mask_pos[mb_inds]) + torch.max(pg_loss1, pg_loss3)*b_mask_pos[mb_inds]).mean()

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
                entropy_mask_loss = (entropy*b_mask_pos[mb_inds]).mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef - args.ent_mask_coef * entropy_mask_loss

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
        # compute shannon entropy and coverage on mu 
        if update % args.shannon_compute_freq == 0:
            shannon_entropy_mu, coverage_mu = env_check.get_shanon_entropy_and_coverage_mu(obs_un)
            wandb.log({"specific/shannon_entropy_mu": shannon_entropy_mu, "specific/coverage_mu": coverage_mu, "global_step": global_step})
            
        # metric
        wandb.log({
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
            "charts/advantages_mean": mb_advantages.mean(),
            "specific/coverage": env_check.get_coverage(),
            "specific/shanon_entropy": env_check.shanon_entropy(),
            "charts/SPS": int(global_step / (time.time() - start_time)),
            "global_step": global_step,
            "update": update,
            "specific/rewards_max": rewards.max().item(),
            "specific/rewards_min": rewards.min().item(),
            "specific/rewards_mean": rewards.mean().item(),
            "specific/rewards_std": rewards.std().item(),
            # LOSS SPECIFIC
            "specific/classification_loss": total_classification_loss,
            "specific/lipshitz_regu": total_lipshitz_regu,
            "specific/beta_adaptive": beta_adaptive,
            "specific/beta_increment": beta_increment,
            "specific/w_rho_un": w_rho_un,
            "specific/rate_w_rho": rate_w_rho,
        })
        if args.make_gif:
            # coverage matrix
            if env_check.matrix_coverage.ndim > 2:
            # Sum over all dimensions except the first two
                reduced_matrix = env_check.matrix_coverage
                for axis in reversed(range(2, reduced_matrix.ndim)):
                    reduced_matrix = np.sum(reduced_matrix, axis=axis)
            else : 
                reduced_matrix = env_check.matrix_coverage
            normalized_matrix = (reduced_matrix - reduced_matrix.min()) / (reduced_matrix.max() - reduced_matrix.min()) * 255
            send_matrix(wandb, np.rot90(normalized_matrix), "coverage", global_step) if update % args.fig_frequency == 0 else None
        # log 
        print('shanon : ', env_check.shanon_entropy())
        print("SPS:", int(global_step / (time.time() - start_time)))
        print(f"global_step={global_step}")
        print('update : ',update)
        print('coverage : ', env_check.get_coverage())  

        if update % args.fig_frequency == 0  and global_step > 0:
            if args.make_gif : 
                image = env_plot.gif(obs_un = obs_un,
                                obs=obs,
                                    classifier = classifier,
                                    device= device)
                send_matrix(wandb, image, "gif", global_step)
            if args.plotly:
                env_plot.plotly(obs_un = obs_un, 
                                classifier = None,
                                device = device)
    if args.save_data:
        # dataset
        send_dataset(wandb, obs_un, actions_un, rewards_un, next_obs_un, dones_un, times_un, "dataset", global_step)
    envs.close()