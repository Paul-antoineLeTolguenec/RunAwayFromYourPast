import numpy as np
import torch


class VE : 
    def __init__(self, n, device, prob) : 
        self.n = n
        self.device = device
        self.prob = prob
        # one-hot encoding
        self.one_hot = torch.eye(n).to(device)
        # index 
        self.z = torch.arange(1,n+1).to(device)
        # probabilities
        self.prob = prob

    def set_prob(self, prob) : 
        self.prob = prob
        
    def sample(self, batch_size, sort=True) : 
        """ sample idx according to the probabilities"""
        batch_idx_z = torch.multinomial(self.prob, batch_size, replacement=True)+1
        # order 
        batch_idx_z, _ = torch.sort(batch_idx_z) if sort else (batch_idx_z, None)
        return batch_idx_z
    def increm(self,z_idx): 
        """ return the increment for the buffer"""
        


if __name__=='__main__': 
    ve = VE(3, 'cpu', torch.tensor([0.1, 0.2, 0.7]))
    print(ve.sample(10))
