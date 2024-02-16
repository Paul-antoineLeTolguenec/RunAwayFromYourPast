from src.utils.replay_buffer import ReplayBuffer
# spectral normalization
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
import torch    

class Classifier(torch.nn.Module):
    def __init__(self, observation_space,device):
        super(Classifier, self).__init__()
        # spectral normalization
        self.fc1 = spectral_norm(torch.nn.Linear(observation_space.shape[0], 128,device=device))
        self.fc2 = spectral_norm(torch.nn.Linear(128, 64, device=device))
        self.fc3 =  spectral_norm(torch.nn.Linear(64, 1, device=device))
        self.relu = torch.nn.ReLU()
        # temperture sigmoid
        self.tau = 1.0
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)/self.tau
    
    def ce_loss(self, batch_q, batch_p):
        return -torch.mean(torch.log(self.sigmoid(self(batch_q))) + torch.log(1 - self.sigmoid(self(batch_p))))
    
    def ce_loss_w_labels(self, batch, labels):
        return -torch.mean(labels*torch.log(self.sigmoid(self(batch))) + (1-labels)*torch.log(1 - self.sigmoid(self(batch))))
    

class Classifier_n(torch.nn.Module):
    def __init__(self, observation_space,device,n_agent, tau = 1.0):
        super(Classifier_n, self).__init__()
        self.n_agent = n_agent  
        # spectral normalization
        self.fc1 = torch.nn.Linear(observation_space.shape[0]+1, 128,device=device)
        self.fc2 = torch.nn.Linear(128, 64, device=device)
        self.fc3 =  torch.nn.Linear(64, 1, device=device)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        # fcz
        self.fcz1 = torch.nn.Linear(observation_space.shape[0], 128,device=device)
        self.fcz2 = torch.nn.Linear(128, 64, device=device)
        self.fcz3 =  torch.nn.Linear(64, n_agent, device=device)
        self.softmax = torch.nn.Softmax(dim=1)
        # temperture sigmoid
        self.tau = tau
        
    def forward(self, x, z):
        x = torch.cat((x,z),1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)/self.tau
    
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

    def ce_loss(self, batch_q, batch_z_q,  batch_p, batch_z_p):
        return -torch.mean(torch.log(self.sigmoid(self(batch_q, batch_z_q))) + torch.log(1 - self.sigmoid(self(batch_p, batch_z_p))))
    
    def ce_loss_w_labels(self, batch, batch_z, labels):
        return -torch.mean(labels*torch.log(self.sigmoid(self(batch, batch_z))) + (1-labels)*torch.log(1 - self.sigmoid(self(batch, batch_z))))
    
    def loss(self, batch_q, batch_z_q,  batch_p, batch_z_p):
        return self.ce_loss(batch_q, batch_z_q,  batch_p, batch_z_p), 0.0
