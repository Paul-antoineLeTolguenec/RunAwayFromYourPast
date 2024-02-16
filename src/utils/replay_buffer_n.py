import numpy as np
import random
import torch

class SampleBatch:
    def __init__(self, observations, actions, next_observations, rewards, dones, z):
        self.observations = observations
        self.actions = actions
        self.next_observations = next_observations
        self.rewards = rewards
        self.dones = dones
        self.z = z


class ReplayBuffer_n:
    def __init__(self, capacity, observation_space, action_space, device, n_agent, probabilities, window_t, handle_timeout_termination=False,):
        self.capacity = capacity
        self.n_env  = n_agent
        self.probabilities = probabilities
        self.pos = np.zeros(n_agent, dtype=int)
        self.full = np.zeros(n_agent, dtype=bool)
        self.device = device
        observation_shape = observation_space.shape
        action_shape = action_space.shape
        # Initialisation des buffers comme arrays NumPy avec la forme appropriée
        self.observations = np.zeros((n_agent, capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((n_agent, capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((n_agent, capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((n_agent, capacity, *observation_shape), dtype=np.float32)
        self.dones = np.zeros((n_agent, capacity, 1), dtype=np.bool_)
        self.window_t = window_t

    def add(self, obs, next_obs, actions, rewards, dones, infos, z_idx, increment,add_pos):
        z_idx = z_idx.cpu().numpy()
        self.observations[z_idx-1, self.pos+increment] = obs.copy()
        self.actions[z_idx-1, self.pos+increment] = actions.copy()
        self.rewards[z_idx-1, self.pos+increment] = np.expand_dims(rewards, axis=1)
        self.next_observations[z_idx-1, self.pos+increment] = next_obs.copy()
        self.dones[z_idx-1, self.pos+increment] = np.expand_dims(dones, axis=1)
        self.pos = (self.pos + add_pos)  % self.capacity
        # check if the buffer is full (np.array)
        self.full = self.full | (self.pos == 0)
        

    def sample(self, batch_size, ve):
        z_idx = ve.sample(batch_size, sort = False).cpu().numpy()
        s_idx = np.random.randint(0, self.pos[z_idx-1])
        # sample from the last window_t
        # s_idx = np.random.randint(self.pos[z_idx-1]-np.ones(batch_size)*self.window_t, self.pos[z_idx-1])
        indices = (z_idx-1, s_idx)
        return self._get_samples(indices,z_idx)

    def sample_threshold(self, t, batch_size,ve):
        z_idx = ve.sample(batch_size, sort = False).cpu().numpy()
        z_idx_random = ve.sample(batch_size, sort = False).cpu().numpy()
        s_idx = np.random.randint(self.pos[z_idx-1]-t, self.pos[z_idx-1])
        s_idx_random = np.random.randint(0, self.pos[z_idx_random-1])
        z_idx_random = ve.sample(batch_size, sort = False).cpu().numpy()
        indices_before_t = (z_idx_random-1, s_idx_random)
        indices_after_t = (z_idx-1, s_idx)
        samples_before_t = self._get_samples(indices_before_t, z_idx_random)
        samples_after_t = self._get_samples(indices_after_t, z_idx)
        return samples_before_t, samples_after_t
    # def sample_threshold(self, t, batch_size,ve):
    #     z_idx = ve.sample(batch_size, sort = False).cpu().numpy()
    #     s_idx = np.random.randint(self.pos[z_idx-1]-t, self.pos[z_idx-1])
    #     s_idx_random = np.random.randint(0, self.pos[z_idx-1]-t)
    #     indices_before_t = (z_idx-1, s_idx_random)
    #     indices_after_t = (z_idx-1, s_idx)

    #     samples_before_t = self._get_samples(indices_before_t, z_idx)
    #     samples_after_t = self._get_samples(indices_after_t, z_idx)
    #     return samples_before_t, samples_after_t
    
    def sample_w_labels(self, t, batch_size,ve):
        z_idx = ve.sample(batch_size, sort = False).cpu().numpy()
        z_idx_random = ve.sample(batch_size, sort = False).cpu().numpy()
        s_idx = np.random.randint(0, self.pos[z_idx-1])
        batch = self._get_samples((z_idx-1, s_idx), z_idx)
        # labels == 1 if self.pos-t<indices and z_idx == z_idx_random
        labels = np.where(((self.pos[z_idx-1]-t) < s_idx ) & (z_idx == z_idx_random), 1, 0)
        # print('labels : ', labels)
        return batch, torch.tensor(labels, dtype=torch.int32, device=self.device)

    def _get_samples(self, indices, z):
        # Utilise l'indexation avancée pour extraire les données et les convertit en tenseurs PyTorch
        return SampleBatch(
            observations=torch.tensor(self.observations[indices[0],indices[1]], dtype=torch.float32, device=self.device),
            actions=torch.tensor(self.actions[indices[0],indices[1]], dtype=torch.float32, device=self.device),
            next_observations=torch.tensor(self.next_observations[indices[0],indices[1]], dtype=torch.float32, device=self.device),
            rewards=torch.tensor(self.rewards[indices[0],indices[1]], dtype=torch.float32, device=self.device),
            dones=torch.tensor(self.dones[indices[0],indices[1]], dtype=torch.float32, device=self.device),
            z=torch.tensor(z, dtype=torch.float32, device=self.device).unsqueeze(1)
        )

    def __len__(self):
        return self.capacity if self.full else self.pos

    def incr_add_pos(self, z_idx):
        unique, increment = np.unique(z_idx, return_counts=True)
        add_pos = np.zeros(self.n_env,dtype=int)
        add_pos[unique-1] = increment
        increment_i = np.concatenate([[j for j in range(increment[i])] for i in range(len(increment))])
        return add_pos, increment_i