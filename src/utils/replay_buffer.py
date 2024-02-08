import numpy as np
import random
import torch

class SampleBatch:
    def __init__(self, observations, actions, next_observations, rewards, dones):
        self.observations = observations
        self.actions = actions
        self.next_observations = next_observations
        self.rewards = rewards
        self.dones = dones


class ReplayBuffer:
    def __init__(self, capacity, observation_space, action_space, device, n_env,  handle_timeout_termination=True,):
        self.capacity = capacity
        self.n_env  = n_env
        self.pos = 0
        self.full = False
        self.device = device
        observation_shape = observation_space.shape
        action_shape = action_space.shape
        # Initialisation des buffers comme arrays NumPy avec la forme appropriée
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.bool_)

    def add(self, obs, next_obs, actions, rewards, dones, infos):
        np.copyto(self.observations[self.pos:self.pos + self.n_env], obs)
        np.copyto(self.actions[self.pos:self.pos + self.n_env], actions)
        np.copyto(self.rewards[self.pos:self.pos + self.n_env], np.expand_dims(rewards, axis=1))
        np.copyto(self.next_observations[self.pos:self.pos + self.n_env], next_obs)
        np.copyto(self.dones[self.pos:self.pos + self.n_env], np.expand_dims(dones, axis=1))
        self.pos = (self.pos + self.n_env) % self.capacity
        self.full = self.full or self.pos == 0
        

    def sample(self, batch_size):
        indices = np.random.randint(0, self.capacity if self.full else self.pos, size=batch_size)
        return self._get_samples(indices)

    def sample_threshold(self, t, batch_size):
        assert 0 <= t <= self.capacity, "Threshold t must be within the range of the buffer."
        
        max_index = self.capacity if self.full else self.pos
        assert batch_size <= t and batch_size <= (max_index - t), "Batch size too large for the specified threshold."

        indices_before_t = np.random.randint(0, t, size=batch_size)
        indices_after_t = np.random.randint(t, max_index, size=batch_size)

        samples_before_t = self._get_samples(indices_before_t)
        samples_after_t = self._get_samples(indices_after_t)

        return samples_before_t, samples_after_t

    def _get_samples(self, indices):
        # Utilise l'indexation avancée pour extraire les données et les convertit en tenseurs PyTorch
        return SampleBatch(
            observations=torch.tensor(self.observations[indices], dtype=torch.float32, device=self.device),
            actions=torch.tensor(self.actions[indices], dtype=torch.float32, device=self.device),
            next_observations=torch.tensor(self.next_observations[indices], dtype=torch.float32, device=self.device),
            rewards=torch.tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
            dones=torch.tensor(self.dones[indices], dtype=torch.float32, device=self.device)
        )

    def __len__(self):
        return self.capacity if self.full else self.pos
