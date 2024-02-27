import numpy as np
import random
import torch
from src.utils.custom_sampling import affine_sample, exp_dec, tau_charac

class SampleBatch:
    def __init__(self, observations, actions, next_observations, rewards, dones, times):
        self.observations = observations
        self.actions = actions
        self.next_observations = next_observations
        self.rewards = rewards
        self.dones = dones
        self.times = times


class ReplayBuffer:
    def __init__(self, capacity, observation_space, action_space, device, n_env, 
                window_t = None ,lambda_rho=2.0, lambda_un=2.0, tau_rho=0.1, tau_un=0.1,
                percentage=0.98, handle_timeout_termination=True,):
        self.capacity = capacity
        self.n_env  = n_env
        self.pos = 0
        self.full = False
        self.device = device
        observation_shape = observation_space.shape
        action_shape = action_space.shape
        self.window_t = window_t
        # Initialisation des buffers comme arrays NumPy avec la forme appropriée
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.bool_)
        self.times= np.zeros((capacity, 1), dtype=np.int32)
        self.lambda_rho = lambda_rho
        self.lambda_un = lambda_un
        # tau_rho between 0 and 0.25
        self.tau_rho = np.clip(tau_rho, 0, 0.25)
        self.tau_un = tau_un
        self.t_safe = int(tau_charac(tau=tau_rho,percentage=percentage)*self.window_t) if window_t is not None else None

    def add(self, obs, next_obs, actions, rewards, dones, infos):
        np.copyto(self.observations[self.pos:self.pos + self.n_env], obs)
        np.copyto(self.actions[self.pos:self.pos + self.n_env], actions)
        np.copyto(self.rewards[self.pos:self.pos + self.n_env], np.expand_dims(rewards, axis=1))
        np.copyto(self.next_observations[self.pos:self.pos + self.n_env], next_obs)
        np.copyto(self.dones[self.pos:self.pos + self.n_env], np.expand_dims(dones, axis=1))
        np.copyto(self.times[self.pos:self.pos + self.n_env], infos['l'])
        self.pos = (self.pos + self.n_env) 
        if self.pos >= self.capacity:
            self.remove_and_shift(self.window_t)
        

    def sample(self, batch_size):
        indices = np.random.randint(0,self.pos-self.t_safe, size=batch_size) if self.window_t is not None else np.random.randint(0,self.pos, size=batch_size)
        # indices = np.random.randint(0,(self.pos-self.t_safe) - (self.pos-self.t_safe)%200, size=batch_size)
        # indices = np.random.randint(self.pos - self.window_t*2 ,(self.pos-self.t_safe) - (self.pos-self.t_safe)%200, size=batch_size)
        return self._get_samples(indices)

    # def sample_threshold(self, t, batch_size):
    #     assert 0 <= t <= self.capacity, "Threshold t must be within the range of the buffer."
    #     max_index = self.capacity if self.full else self.pos
    #     # assert batch_size <= t and batch_size <= (max_index - t), "Batch size too large for the specified threshold."
    #     indices_before_t = np.random.randint(0, max_index-t, size=batch_size)
    #     indices_after_t = np.random.randint(max_index-t, max_index, size=batch_size)
    #     # indices_after_t = np.random.randint(0, max_index, size=batch_size)
    #     samples_before_t = self._get_samples(indices_before_t)
    #     samples_after_t = self._get_samples(indices_after_t)
    #     return samples_before_t, samples_after_t

    def sample_threshold(self, t, batch_size):
        assert 0 <= t <= self.capacity, "Threshold t must be within the range of the buffer."
        max_index = self.capacity if self.full else self.pos
        # sample affine transformation of the uniform distribution
        # dist_rho = affine_sample(batch_size, self.lambda_rho)
        # sample exponential transformation of the uniform distribution
        dist_rho = exp_dec(batch_size, self.tau_rho)
        # dist_rho = np.random.rand(batch_size)
        # dist_un = affine_sample(batch_size, self.lambda_un)
        dist_un = exp_dec(batch_size, self.tau_un)
        # dist_un = np.random.rand(batch_size)
        # convert indices 
        indices_rho = (dist_rho * t).astype(int)+int(max_index-t)
        # indices_rho = (dist_rho * max_index).astype(int)
        indices_un = (dist_un * (max_index-t)).astype(int)
        # indices_un = (dist_un * 4*t).astype(int) + int(max_index-5*t)
        # print('min max indices', np.min(indices_rho), np.max(indices_rho), np.min(indices_un), np.max(indices_un))
        # indices_after_t = np.random.randint(0, max_index, size=batch_size)
        samples_before_t = self._get_samples(indices_un)
        samples_after_t = self._get_samples(indices_rho)
        return samples_before_t, samples_after_t
    
    # def sample_w_labels(self, t, batch_size):
    #     max_index = self.capacity if self.full else self.pos
    #     indices = np.random.randint(0, max_index, size=batch_size)
    #     batch = self._get_samples(indices)
    #     # labels == 1 if self.pos-t<indices else 0 without for loop
    #     labels = np.where((self.pos-t) < indices, 1, 0)
    #     return batch, torch.tensor(labels, dtype=torch.float32, device=self.device)

    def sample_w_labels(self, t, batch_size):
        max_index = self.capacity if self.full else self.pos
        dist = affine_sample(batch_size, self.lambda_un)
        indices = (dist * (max_index)).astype(int)
        batch = self._get_samples(indices)
        # labels == 1 if self.pos-t<indices else 0 without for loop
        labels = np.where((self.pos-t) < indices, 1, 0)
        return batch, torch.tensor(labels, dtype=torch.float32, device=self.device)

    def _get_samples(self, indices):
        # Utilise l'indexation avancée pour extraire les données et les convertit en tenseurs PyTorch
        return SampleBatch(
            observations=torch.tensor(self.observations[indices], dtype=torch.float32, device=self.device),
            actions=torch.tensor(self.actions[indices], dtype=torch.float32, device=self.device),
            next_observations=torch.tensor(self.next_observations[indices], dtype=torch.float32, device=self.device),
            rewards=torch.tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
            dones=torch.tensor(self.dones[indices], dtype=torch.float32, device=self.device),
            times=torch.tensor(self.times[indices], dtype=torch.int32, device=self.device)
        )

    def __len__(self):
        return self.capacity if self.full else self.pos


    def remove_and_shift(self, t, dist = None) -> int:
        # Randomly selects an index between 0 and t and select from probability distribution if provided
        idx_to_remove = np.random.randint(0, self.pos) if dist is None else np.random.choice(self.pos, p=dist)

        # Shifts elements after the removed index one position to the left
        self.observations = np.delete(self.observations, idx_to_remove, axis=0)
        self.actions = np.delete(self.actions, idx_to_remove, axis=0)
        self.rewards = np.delete(self.rewards, idx_to_remove, axis=0)
        self.next_observations = np.delete(self.next_observations, idx_to_remove, axis=0)
        self.dones = np.delete(self.dones, idx_to_remove, axis=0)
        # add zeros at the end to keep the same shape
        self.observations = np.vstack((self.observations, np.zeros((1, *self.observations.shape[1:]))))
        self.actions = np.vstack((self.actions, np.zeros((1, *self.actions.shape[1:]))))
        self.rewards = np.vstack((self.rewards, np.zeros((1, *self.rewards.shape[1:]))))
        self.next_observations = np.vstack((self.next_observations, np.zeros((1, *self.next_observations.shape[1:]))))
        self.dones = np.vstack((self.dones, np.zeros((1, *self.dones.shape[1:]))))
        # Decreases the index t by 1, as an element has been removed
        self.pos -= 1

    def full_rb_batch(self) : 
        return SampleBatch(
            observations=torch.tensor(self.observations[:self.pos], dtype=torch.float32, device=self.device),
            actions=torch.tensor(self.actions[:self.pos], dtype=torch.float32, device=self.device),
            next_observations=torch.tensor(self.next_observations[:self.pos], dtype=torch.float32, device=self.device),
            rewards=torch.tensor(self.rewards[:self.pos], dtype=torch.float32, device=self.device),
            dones=torch.tensor(self.dones[:self.pos], dtype=torch.float32, device=self.device),
            times=torch.tensor(self.times[:self.pos], dtype=torch.int32, device=self.device)
        )