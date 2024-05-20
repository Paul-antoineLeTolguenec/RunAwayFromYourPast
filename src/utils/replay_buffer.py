import numpy as np
import random
import torch
import numpy as np
import socket, subprocess

class SampleBatch:
    def __init__(self, observations, actions, next_observations, rewards, dones, times):
        self.observations = observations
        self.actions = actions
        self.next_observations = next_observations
        self.rewards = rewards
        self.dones = dones
        self.times = times


class ReplayBuffer:
    def __init__(self, 
                 capacity, 
                 observation_space, 
                 action_space, 
                 device, 
                 run_init_path = None):
        self.capacity = capacity
        self.pos = 0
        self.full = False
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space
        observation_shape = observation_space.shape
        action_shape = action_space.shape
        # Initialisation des buffers comme arrays NumPy avec la forme appropriée
        if run_init_path is not None:
            self.observations, self.actions, self.next_observations, self.rewards, self.dones, self.times = self.load(run_init_path)
            self.pos = len(self.observations)
            self.full = True if self.pos >= self.capacity else False
        else:
            self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
            self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
            self.rewards = np.zeros((capacity, 1), dtype=np.float32)
            self.next_observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
            self.dones = np.zeros((capacity, 1), dtype=np.bool_)
            self.times = np.zeros((capacity, 1), dtype=np.int32)
        

    def add(self, obs, next_obs, actions, rewards, dones, infos):
        np.copyto(self.observations[self.pos:self.pos + self.n_env], obs)
        np.copyto(self.actions[self.pos:self.pos + self.n_env], actions)
        np.copyto(self.rewards[self.pos:self.pos + self.n_env], np.expand_dims(rewards, axis=1))
        np.copyto(self.next_observations[self.pos:self.pos + self.n_env], next_obs)
        np.copyto(self.dones[self.pos:self.pos + self.n_env], np.expand_dims(dones, axis=1))
        self.pos = (self.pos + self.n_env) 
        if self.pos >= self.capacity:
            self.remove_and_shift(self.window_t)
        

    def sample(self, batch_size, pos_rho = 0):
        indices = np.random.randint(pos_rho,self.pos, size=batch_size) 
        return self._get_samples(indices)

   

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
    

    def save(self, run_id, obs, actions, next_obs, rewards, dones, infos):
        # check HOSTNAME
        if 'bubo' in socket.gethostname():
            path_data = 'dataset/'
        elif 'pando' in socket.gethostname():
            path_data = '/scratch/disc/p.le-tolguenec/dataset/'
        elif 'olympe' in socket.gethostname():
            path_data = '/tmpdir/'+subprocess.run(['whoami'], stdout=subprocess.PIPE, text=True).stdout.strip()+'/dataset/'
        # save the dataset
        np.savez(path_data+'dataset_'+str(run_id)+'.npz', obs=obs, actions=actions, next_obs=next_obs, rewards=rewards, dones=dones, infos=infos)


    def load(self, algo, env, seed) : 
        # check HOSTNAME
        if 'bubo' in socket.gethostname():
            path_data = 'dataset/'
        elif 'pando' in socket.gethostname():
            path_data = '/scratch/disc/p.le-tolguenec/dataset/'
        elif 'olympe' in socket.gethostname():
            path_data = '/tmpdir/'+subprocess.run(['whoami'], stdout=subprocess.PIPE, text=True).stdout.strip()+'/dataset/'
        # load the dataset
        data = np.load(path_data+'dataset_'+env+'__'+algo+'__'+str(seed)+'.npz')
        return data['obs'], data['actions'], data['next_obs'], data['rewards'], data['dones'], data['infos']
