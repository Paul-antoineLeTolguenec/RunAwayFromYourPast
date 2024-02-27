from src.utils.replay_buffer import ReplayBuffer
# spectral normalization
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
import torch    

class Classifier(torch.nn.Module):
    def __init__(self, observation_space,device, env_max_steps, 
                lipshitz= False, lim_down = -10, lim_up = 10, 
                treshold_old = -5, w_old = 0.01):
        super(Classifier, self).__init__()
        # spectral normalization
        self.fc1 = spectral_norm(torch.nn.Linear(observation_space.shape[0], 128,device=device)) if lipshitz else torch.nn.Linear(observation_space.shape[0], 128,device=device)
        self.fc2 = spectral_norm(torch.nn.Linear(128, 64, device=device)) if lipshitz else torch.nn.Linear(128, 64, device=device)
        self.fc3 =  spectral_norm(torch.nn.Linear(64, 1, device=device)) if lipshitz else torch.nn.Linear(64, 1, device=device)
        self.relu = torch.nn.ReLU()
        self.env_max_steps = env_max_steps
        self.lim_down = lim_down
        self.lim_up = lim_up
        self.w_old = w_old
        self.treshold_old = treshold_old
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        # return x
        return torch.clamp(x, self.lim_down, self.lim_up)
    
    # def ce_loss(self, batch_q, batch_p):
    #     relabeling = self.relabeling(batch_q.times, self.env_max_steps)
    #     s_q = self.sigmoid(self(batch_q.observations))
    #     s_p = self.sigmoid(self(batch_p.observations))
    #     # mask s_q s_q inf or equal to trehold_old 
    #     mask_q = (s_q <= self.treshold_old).float()
    #     label_q = torch.ones_like(s_q) - (1-self.w_old)*mask_q
    #     label_q = self.relabeling(batch_q.times, self.env_max_steps) * label_q
    #     return -torch.mean(label_q*torch.log(s_q) + torch.log(1 - s_p))
        # return -torch.mean(torch.log(self.sigmoid(self(batch_q.observations))) + torch.log(1 - self.sigmoid(self(batch_p.observations))))

    def ce_loss_ppo(self, batch_q, times_q, batch_p, relabeling = True):
        s_q = self.sigmoid(self(batch_q))
        s_p = self.sigmoid(self(batch_p))
        # mask s_q s_q inf or equal to trehold_old 
        if relabeling : 
            mask_q = (s_q <= self.treshold_old).float()
            label_q = torch.ones_like(s_q) - (1-self.w_old)*mask_q
            label_q = self.relabeling(times_q, self.env_max_steps) * label_q
            return -torch.mean(label_q*torch.log(s_q) + torch.log(1 - s_p))
        else :
            return -torch.mean(torch.log(s_q) + torch.log(1 - s_p))
    

    # def ce_loss_w_labels(self, batch, labels):
    #     return -torch.mean(labels*torch.log(self.sigmoid(self(batch))) + (1-labels)*torch.log(1 - self.sigmoid(self(batch))))
    
    # def relabeling(self, t, max_steps, percentage=0.5):
    #     """ Response of order 1 """
    #     # 95 %
    #     tau = -max_steps / np.log(1 - percentage)
    #     return (1-torch.exp(-t / tau))
    
    def relabeling(self, t, max_steps, tau=3.0):
        """ exp(T)=1 
            tau in ]0,10] """
        return torch.exp((t-max_steps)/tau)
    

class Classifier_n(torch.nn.Module):
    def __init__(self, observation_space,device,n_agent,env_max_steps, 
                lipshitz= False, lim_down = -10, lim_up = 10, 
                treshold_old = -5, w_old = 0.05):
        super(Classifier_n, self).__init__()
        self.n_agent = n_agent  
        # spectral normalization
        self.fc1 = spectral_norm(torch.nn.Linear(observation_space.shape[0]+1, 128,device=device)) if lipshitz else torch.nn.Linear(observation_space.shape[0]+1, 128,device=device)
        self.fc2 =spectral_norm( torch.nn.Linear(128, 64, device=device)) if lipshitz else torch.nn.Linear(128, 64, device=device)
        self.fc3 = spectral_norm(torch.nn.Linear(64, 1, device=device)) if lipshitz else torch.nn.Linear(64, 1, device=device)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        # # fcz
        self.fcz1 = torch.nn.Linear(observation_space.shape[0], 128,device=device)
        self.fcz2 = torch.nn.Linear(128, 64, device=device)
        self.fcz3 =  torch.nn.Linear(64, n_agent, device=device)
        self.softmax = torch.nn.Softmax(dim=1)
        self.env_max_steps = env_max_steps
        self.lim_down = lim_down
        self.lim_up = lim_up
        self.w_old = w_old
        self.treshold_old = treshold_old
        
    def forward(self, x, z):
        x = torch.cat((x,z),-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # return self.fc3(x)
        return torch.clamp(self.fc3(x), self.lim_down, self.lim_up)
    
    def forward_z(self, x):
        x = self.relu(self.fcz1(x))
        x = self.relu(self.fcz2(x))
        return self.fcz3(x)
    
    def mlh_loss(self, obs, z):
        # change dtype to int
        z = z.type(torch.int64)-1
        p_z = self.softmax(self.forward_z(obs))
        p_z_i = torch.gather(p_z, 1, z)
        return -torch.mean(torch.log(p_z_i))


    def relabeling(self, t, max_steps, tau=5.0):
        """ exp(T)=1 
            tau in ]0,10] """
        return torch.exp((t-max_steps)/tau)
    
    # def ce_loss(self, batch_q,  batch_p):
    #     relabeling = self.relabeling(batch_q.times, self.env_max_steps)
    #     s_q = self.sigmoid(self(batch_q.observations, batch_q.z))
    #     s_p = self.sigmoid(self(batch_p.observations, batch_p.z))
    #     # mask s_q s_q inf or equal to trehold_old 
    #     mask_q = (s_q <= self.treshold_old).float()
    #     label_q = torch.ones_like(s_q) - (1-self.w_old)*mask_q
    #     label_q = self.relabeling(batch_q.times, self.env_max_steps) * label_q
    #     return -torch.mean(label_q*torch.log(s_q) + torch.log(1 - s_p))

    def ce_loss_ppo(self, batch_q_s,  batch_q_z, times_q,  batch_p_s, batch_p_z,relabeling = True):
        s_q = self.sigmoid(self(batch_q_s, batch_q_z))
        s_p = self.sigmoid(self(batch_p_s, batch_p_z))
        if relabeling : 
            mask_q = (s_q <= self.treshold_old).float()
            label_q = torch.ones_like(s_q) - (1-self.w_old)*mask_q
            # label_q = self.relabeling(times_q, self.env_max_steps) * label_q
            return -torch.mean(label_q*torch.log(s_q) + torch.log(1 - s_p))
        else :
            return -torch.mean(torch.log(s_q) + torch.log(1 - s_p))
    
    # def ce_loss_w_labels(self, batch, batch_z, labels):
    #     return -torch.mean(labels*torch.log(self.sigmoid(self(batch, batch_z))) + (1-labels)*torch.log(1 - self.sigmoid(self(batch, batch_z))))
    
    def loss(self, batch_q,  batch_p, batch_z):
        return self.ce_loss(batch_q, batch_p), self.mlh_loss(batch_z.observations, batch_z.z)
