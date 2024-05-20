import torch 
import torch.nn as nn



class WorldModel(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(WorldModel, self).__init__()
        self.fc1 = nn.Linear(observation_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.f_mean = nn.Linear(128, observation_dim)
        self.f_logstd = nn.Linear(128, observation_dim)
        self.relu = nn.ReLU()

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = self.f_mean(x)
        logstd = self.f_logstd(x)
        return mean, logstd
    
    def mlh_loss(self, obs, a, next_obs):
        mean, logstd = self.forward(obs, a)
        dist = torch.distributions.Normal(mean, torch.exp(logstd))
        return -dist.log_prob(next_obs).mean()
    
    def empirical_risk(self, obs, a, next_obs):
        mean, logstd = self.forward(obs, a)
        return ((next_obs - mean)**2).sum(dim=1).mean()