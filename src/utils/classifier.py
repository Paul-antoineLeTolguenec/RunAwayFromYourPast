import torch


class Classifier(torch.nn.Module):
    def __init__(self, observation_space,n_agent, device):
        super(Classifier, self).__init__()
        self.fc1 = torch.nn.Linear(observation_space, 128, device=device)
        self.fc2 = torch.nn.Linear(128, n_agent, device=device)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def mlh_loss(self, obs, z):
        # change dtype to int
        z = z.type(torch.int64)-1
        p_z = self.softmax(self.forward(obs))
        p_z_i = torch.gather(p_z, 1, z)
        return -torch.mean(torch.log(p_z_i))