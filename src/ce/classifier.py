from src.utils.replay_buffer import ReplayBuffer
import numpy as np
import torch    

class Classifier(torch.nn.Module):
    def __init__(self, observation_space,device):
        super(Classifier, self).__init__()
        self.fc1 = torch.nn.Linear(observation_space.shape[0], 128,device=device)
        self.fc2 = torch.nn.Linear(128, 64, device=device)
        self.fc3 = torch.nn.Linear(64, 1, device=device)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    
    def ce_loss(self, batch_q, batch_p):
        return -torch.mean(torch.log(self.sigmoid(self(batch_q))) + torch.log(1 - self.sigmoid(self(batch_p))))