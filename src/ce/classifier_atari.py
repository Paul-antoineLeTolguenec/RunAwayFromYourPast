from src.utils.replay_buffer import ReplayBuffer
from envs.config_env import config
# spectral normalization
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
import torch.nn as nn
import torch

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ClassifierAtari(torch.nn.Module):
    def __init__(self, 
                observation_space,
                device, 
                lim_down = -10, 
                lim_up = 10, 
                learn_z = False, 
                n_agent = 1, 
                env_id = None,
                feature_extractor = False, 
                use_lstm = False):
        super(ClassifierAtari, self).__init__()
        if feature_extractor:
            self.fc1 = torch.nn.Linear(config[env_id]['coverage_idx'].shape[0], 128,device=device)
        else:
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
            # if use_lstm:
            #     self.fc1 = nn.LSTM(512, 128).to(device)
            #     for name, param in self.fc1.named_parameters():
            #         if "bias" in name:
            #             nn.init.constant_(param, 0)
            #         elif "weight" in name:
            #             nn.init.orthogonal_(param, 1.0)
            # else :
            self.fc1 = torch.nn.Linear(512, 128,device=device)
        # predictor
        self.predictor = layer_init(nn.Linear(128, 1), std=0.01)
        
        self.relu = torch.nn.ReLU()
        self.lim_down = lim_down
        self.lim_up = lim_up
        self.sigmoid = torch.nn.Sigmoid()
        self.learn_z = learn_z
        self.n_agent = n_agent
        self.feature_extractor = feature_extractor
        self.env_id = env_id
        if learn_z : 
            if feature_extractor:
                self.fcz1 = torch.nn.Linear(config[env_id]['coverage_idx'].shape[0], 128,device=device)
            else:
                self.cnn_z = nn.Sequential(
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
                self.fcz1 = torch.nn.Linear(512, 128,device=device)
            self.predictor_z = layer_init(nn.Linear(128, n_agent), std=0.01)
            self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        # NOT LSTM FOR NOW
        # x = self.feature(x) if self.feature_extractor else x
        x = self.relu(self.fc1(self.cnn(x)))
        x = self.predictor(x)
        return torch.clamp(x, self.lim_down, self.lim_up)
    
    def forward_z(self, x):
        x = self.feature(x) if self.feature_extractor else x
        x = self.relu(self.fcz1(x)) if self.feature_extractor else self.cnn_z(x)
        x = self.predictor_z(x)
        return x
    
    def mlh_loss(self, obs, z):
        # change dtype to int
        z = z.type(torch.int64)-1
        p_z = self.softmax(self.forward_z(obs))
        p_z_i = torch.gather(p_z, 1, z)
        return -torch.mean(torch.log(p_z_i))
    

    def ce_loss_ppo(self, batch_q, batch_p, batch_q_z = None, batch_p_z =None, relabeling = True, ratio = 1.0, return_log_and_prob = False):
        s_q = self(batch_q)
        s_q_p = self.sigmoid(s_q)
        s_p = self(batch_p)
        s_p_p = self.sigmoid(s_p)
        # mask strategy q
        label_q = torch.ones_like(s_q)
        # mask strategy p
        label_p = torch.ones_like(s_p)
        L = -((label_q*torch.log(s_q_p)) +(label_p*torch.log(1 - s_p_p))).mean()
        # if self.learn_z:
        #     L += self.mlh_loss(batch_q, batch_q_z) + self.mlh_loss(batch_p, batch_p_z)
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
    