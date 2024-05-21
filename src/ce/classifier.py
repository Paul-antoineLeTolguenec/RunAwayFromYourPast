from src.utils.replay_buffer import ReplayBuffer
from envs.config_env import config
# spectral normalization
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
import torch    


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Classifier(torch.nn.Module):
    def __init__(self, observation_space,device, env_max_steps, 
                lipshitz= False, lim_down = -10, lim_up = 10, 
                w_old = 0.0001, learn_z = False, 
                n_agent = 1, n_reconf = 0, 
                env_id = None,
                bound_spectral = 1,
                iter_lip = 1,
                feature_extractor = False,
                lipshitz_regu = False,
                lambda_init = 30.0, 
                lip_cte = 1.0,
                epsilon = 1e-3):
        super(Classifier, self).__init__()
        if feature_extractor:
            self.fc1 = torch.nn.Linear(config[env_id]['coverage_idx'].shape[0], 128,device=device)
            self.fc2 = torch.nn.Linear(128, 64, device=device)
            self.fc3 = torch.nn.Linear(64, 1, device=device)
        elif lipshitz:
            self.fc1 = spectral_norm(layer_init(torch.nn.Linear(observation_space.shape[0], 128)).to(device), n_power_iterations =iter_lip)
            self.fc2 = spectral_norm(layer_init(torch.nn.Linear(128, 64)).to(device), n_power_iterations =iter_lip)
            self.fc3 = spectral_norm(layer_init(torch.nn.Linear(64, 1)).to(device), n_power_iterations =iter_lip)
        else:
            self.fc1 = layer_init(torch.nn.Linear(observation_space.shape[0], 128)).to(device)
            self.fc2 = layer_init(torch.nn.Linear(128, 64)).to(device)
            self.fc3 = layer_init(torch.nn.Linear(64, 1)).to(device)
            # self.fc1 =torch.nn.Linear(observation_space.shape[0], 128).to(device)
            # self.fc2 =torch.nn.Linear(128, 64).to(device)
            # self.fc3 =torch.nn.Linear(64, 1).to(device)
        self.lambda_lip = torch.nn.Parameter(torch.tensor(lambda_init, device=device))
        self.epsilon = torch.tensor(epsilon, device=device)
        self.lipshitz_regu = lipshitz_regu
        self.relu = torch.nn.ReLU()
        self.env_max_steps = env_max_steps
        self.lim_down = lim_down
        self.lim_up = lim_up
        self.w_old = w_old
        self.sigmoid = torch.nn.Sigmoid()
        self.learn_z = learn_z
        self.n_agent = n_agent
        self.n_reconf = n_reconf
        self.feature_extractor = feature_extractor
        self.lipshitz = lipshitz
        self.bound_spectral = bound_spectral
        self.env_id = env_id
        self.lip_cte = lip_cte
        if learn_z : 
            if feature_extractor:
                self.fcz1 = torch.nn.Linear(config[env_id]['coverage_idx'].shape[0], 128,device=device)
                self.fcz2 = torch.nn.Linear(128, 64, device=device)
                self.fcz3 =  torch.nn.Linear(64, n_agent, device=device)
            else:
                self.fcz1 = torch.nn.Linear(observation_space.shape[0], 128,device=device)
                self.fcz2 = torch.nn.Linear(128, 64, device=device)
                self.fcz3 =  torch.nn.Linear(64, n_agent, device=device)
            self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.feature(x) if self.feature_extractor else x
        x = self.relu(self.fc1(x))
        x = self.bound_spectral * x if self.lipshitz else x
        x = self.relu(self.fc2(x))
        x = self.bound_spectral * x if self.lipshitz else x
        x = self.fc3(x)
        x = self.bound_spectral * x if self.lipshitz else x
        # return x
        return torch.clamp(x, self.lim_down, self.lim_up)
    
    def forward_z(self, x):
        x = self.feature(x) if self.feature_extractor else x
        x = self.relu(self.fcz1(x))
        x = self.relu(self.fcz2(x))
        return self.fcz3(x)
    
    def mlh_loss(self, obs, z):
        # change dtype to int
        p_z = (self.softmax(self.forward_z(obs))*z).sum(dim=-1)
        return -torch.mean(torch.log(p_z))
    

    def ce_loss_ppo(self, batch_q, batch_p, 
                    batch_s_q = None, batch_s_p = None,
                    batch_q_z = None, batch_p_z =None, 
                    beta = 0.5):
        s_q = self(batch_q)
        s_q_p = self.sigmoid(s_q)
        s_p = self(batch_p)
        s_p_p = self.sigmoid(s_p)
        # mask strategy q
        label_q = torch.ones_like(s_q)
        # mask strategy p
        label_p = torch.ones_like(s_p)
        L = -((label_q*torch.log(s_q_p)).mean() +(label_p*torch.log(1 - s_p_p)).mean()) 
        if self.learn_z:
            L += beta*self.mlh_loss(batch_s_q, batch_q_z) + (1-beta)*self.mlh_loss(batch_s_p, batch_p_z)
        return L 
       


    def mask_labels_q(self, s_q, tau=0.5): #1.0
        with torch.no_grad():
            s_q_clip = torch.clamp(s_q, self.lim_down, 0)
            label_q = torch.exp(s_q_clip/(-self.lim_down*tau))
            return label_q
       
    def mask_labels_p(self, s_p,w=1.0): #1.0
        with torch.no_grad():
            mask_p = (0.0 <= s_p).float()
            label_p = torch.ones_like(s_p) + mask_p*w
            return label_p
        
    def feature(self, x):
        x=  x[:, :, config[self.env_id]['coverage_idx']] if x.dim() == 3 else x[:, config[self.env_id]['coverage_idx']]
        return x
    
    def lipshits_regu(self, obs, next_obs, dones):
        o = self(obs)
        next_o = self(next_obs)
        L = (torch.min(self.epsilon,self.lip_cte-torch.norm(o-next_o, dim=-1))*(1-dones))
        return -L.mean()

    def lipshitz_loss_ppo(self, 
                        batch_q, batch_p, 
                        q_batch_s = None, q_batch_next_s = None, q_dones = None,
                        p_batch_s = None, p_batch_next_s = None, p_dones = None, 
                        ratio = 1.0):
        s_q = self(batch_q)
        s_q_p = self.sigmoid(s_q)
        s_p = self(batch_p)
        s_p_p = self.sigmoid(s_p)
        # mask strategy q
        label_q = torch.ones_like(s_q)
        # mask strategy p
        label_p = torch.ones_like(s_p)
        # classification loss
        # L = -((label_q*torch.log(s_q_p)) +(label_p*torch.log(1 - s_p_p))).mean() 
        L =-((s_q - s_p)).mean()
        # lipshitz regularization
        lipshitz_loss = (self.lipshits_regu(q_batch_s, q_batch_next_s, q_dones.squeeze(-1)) + self.lipshits_regu(p_batch_s, p_batch_next_s, p_dones))
        # if self.learn_z:
        #     L += self.mlh_loss(batch_q, batch_q_z) + self.mlh_loss(batch_p, batch_p_z)
        return L+lipshitz_loss*self.lambda_lip.detach(), -lipshitz_loss.detach()