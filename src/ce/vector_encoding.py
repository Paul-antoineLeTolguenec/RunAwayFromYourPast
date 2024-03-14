import numpy as np
import torch


class VE : 
    def __init__(self, n, device, prob, n_reconf = 0) : 
        self.n = n
        self.device = device
        self.prob = prob
        # index 
        self.z = torch.arange(1,n+1).to(device) if n_reconf == 0 else torch.arange(1,n+1).unsqueeze(-1).repeat(1,n_reconf).to(device)
        # probabilities
        self.prob = prob


    def set_prob(self, prob) : 
        self.prob = prob
        
    def sample(self, batch_size, sort=True, uniform = False) : 
        """ sample idx according to the probabilities"""
        batch_idx_z = torch.multinomial(self.prob, batch_size, replacement=True)+1 if not uniform else torch.randint(1, self.n+1, (batch_size,))
        # order 
        batch_idx_z, _ = torch.sort(batch_idx_z) if sort else (batch_idx_z, None)
        return batch_idx_z
    
    def shuffle(self, z, epoch) : 
        """ shuffle z for column greater than epoch"""
        permutations = torch.randperm(self.n).to(self.device)
        save_z = z[:,:epoch+1]
        z = torch.cat((save_z, z[permutations,epoch+1:]), dim = 1)
        return z
if __name__=='__main__': 
    ve = VE(3, 'cpu', 1/3, 10)
    print('self.')
    print('z shape : ', ve.z.shape)
    print('z : ', ve.z)
    print('shuffle z : ', ve.shuffle(ve.z, 1))
