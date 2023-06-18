import matplotlib.pyplot as plt
import numpy as np
import gym, torch, os , imageio, socket, subprocess
from src.utils.dash_utils_2d import initialize_figure_2d, add_frame_to_figure_2d, create_html_2d
from src.utils.dash_utils_3d import initialize_figure_3d, add_frame_to_figure_3d, create_html_3d
from envs.compatible_random_generator import CompatibleRandomGenerator
import gymnasium
import sys, signal
from functools import reduce
from operator import mul

class Wenv(gym.Env):
    def __init__(self, env_id , 
                 kwargs = {},
                 coverage_idx = np.array([0,1]),
                 coverage_accuracy=100, 
                 render_bool_matplot = False,
                 render_bool_plotly = False,
                 plotly_sample_per_episode = 2000,
                 render_settings = {'x_lim': [-1, 1], 'y_lim': [-1, 1]},
                 xp_id = None, 
                 type_id = 'Maze'):
        super(Wenv, self).__init__()
        self.env_id = env_id
        self.xp_id = xp_id
        self.type_id = type_id
        self.config = kwargs
        # load the environment
        self.env = gym.make(env_id, **kwargs)
        # observation space
        self.observation_space = self.env.observation_space if not type_id == 'robotics' else self.env.observation_space['observation']
        # action space
        self.action_space = self.env.action_space
        # spec 
        self.spec = self.env.spec
        # metrics
        self.coverage_accuracy = coverage_accuracy
        self.matrix_coverage = np.zeros((self.coverage_accuracy,)*coverage_idx.shape[0])
        self.coverage_idx = coverage_idx
        self.rooms = []
        self.episode_length = 0
        self.episode_reward = 0
        self.limits = list(render_settings.values())
        self.transposed_limits = np.array(self.limits).transpose()
        # figure 
        if render_bool_matplot:
            # check HOSTNAME
            if 'bubo' in socket.gethostname():
                self.path_gif = 'gif/'
            elif 'pando' in socket.gethostname():
                self.path_gif = '/scratch/disc/p.le-tolguenec/gif/'
            elif 'olympe' in socket.gethostname():
                self.path_gif = '/tmpdir/'+subprocess.run(['whoami'], stdout=subprocess.PIPE, text=True).stdout.strip()+'/gif/'
            # SET UP
            self.figure, self.ax = plt.subplots()
            self.render_settings = render_settings
            self.ax.set_xlim([render_settings['x_lim'][0], render_settings['x_lim'][1]])
            self.ax.set_ylim([render_settings['y_lim'][0], render_settings['y_lim'][1]])
            self.writer_gif = self.set_gif(xp_id)
            self.obs_saved = []
            self.random_colors = None
            
        # plotly
        if render_bool_plotly:
            self.frame = 0
            self.plotly_sample_per_episode = plotly_sample_per_episode
            signal.signal(signal.SIGINT, self.signal_handler)
            if self.coverage_idx.shape[0] == 2:
                self.figure = initialize_figure_2d(render_settings=render_settings)
            if self.coverage_idx.shape[0] == 3:
                self.figure = initialize_figure_3d(render_settings=render_settings)
        # specific for atari
        if type_id == 'atari':
            original_generator = self.env.unwrapped.np_random
            compatible_generator = CompatibleRandomGenerator(original_generator)
            self.env.unwrapped.np_random = compatible_generator
            self.observation_space = gymnasium.spaces.Box(self.env.observation_space.low, self.env.observation_space.high, dtype=np.uint8)
            self.action_space = gymnasium.spaces.Discrete(self.action_space.n)
            self.render_mode = kwargs['render_mode']


    def set_all_seeds(self, seed):
        self.seed(seed)
        self.env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def reset(self, seed = 0, options = None):
        # self.seed(seed)
        self.episode_length = 0
        self.episode_reward = 0
        obs, i = self.env.reset()
        if self.type_id == 'robotics':
           obs = self.parser_robotics(obs)
        elif self.type_id == 'atari' : 
            obs, i = self.env.reset()
            i['position'] = self.parser_ram_atari()
        # update episode length and reward
        i['l'] = self.episode_length
        i['r'] = self.episode_reward
        return obs, i #if not self.type_id == 'atari' else obs
    

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        self.episode_length += 1
        self.episode_reward += reward
        if self.type_id == 'robotics':
            obs = self.parser_robotics(obs)
        elif self.type_id == 'atari':
            info['position'] = self.parser_ram_atari()
        # update episode length and reward
        info['l'] = self.episode_length
        info['r'] = self.episode_reward
        return obs, reward, done, trunc, info #if not self.type_id == 'atari' else obs, reward, done, info
    
    def render(self):
        return self.env.render()
    def close(self):
        return self.env.close()
    def seed(self, seed=None):
        self.env.seed = seed
    def __str__(self):
        return self.env.__str__()
    
    def gif(self, obs_un=None, obs_un_train=None, 
            obs=None, classifier=None, 
            device=None , z_un = None, 
            zs = None, obs_rho = None, 
            adv = None, obs_adv = None,):
        # self.random_colors = [ np.random.rand(3,) for _ in range(z_un.shape[-1])] if (z_un is not None and self.random_colors is None) else self.random_colors
        self.random_colors = [ np.random.rand(3,) for _ in range(zs.shape[-1])] if (zs is not None and self.random_colors is None) else self.random_colors
        if adv is not None:
            self.ax.clear()
            sc = self.ax.scatter(obs_adv[:,self.coverage_idx[0]], obs_adv[:,self.coverage_idx[1]], s=1, c=adv, cmap='viridis')
            if hasattr(self, 'cbar'):
                self.cbar.update_normal(sc)
            else:
                self.cbar = self.figure.colorbar(sc, ax=self.ax)

        elif classifier is not None:
            with torch.no_grad():
                # clear the plot
                self.ax.clear()
                # data  to plot
                # data_to_plot =torch.Tensor(obs_un.reshape(-1, *self.observation_space.shape)).to(device)
                data_to_plot =torch.cat([torch.Tensor(obs_un.reshape(-1, *self.observation_space.shape)).to(device), torch.Tensor(obs.reshape(-1, *self.observation_space.shape))], dim=0)
                # Plotting measure 
                m_n = classifier(data_to_plot)
                # mask =torch.nonzero(m_n> 0, as_tuple=True)[0]
                # mask_z =torch.nonzero(m_n< 0, as_tuple=True)[0]
                m_n = m_n.detach().cpu().numpy().squeeze(-1)
                # normalize
                # m_n = (m_n - m_n.mean()) / m_n.std()
                # data to plot
                data_to_plot = data_to_plot.detach().cpu().numpy()
                # Plotting the environment
                # self.ax.scatter(data_to_plot[:,self.coverage_idx[0]], data_to_plot[:,self.coverage_idx[1]], s=1, c = m_n, cmap='viridis')
                # plot obs train
                # self.ax.scatter(data_to_plot[mask,self.coverage_idx[0]], data_to_plot[mask,self.coverage_idx[1]], s=1, c='g')
                # self.ax.scatter(data_to_plot[mask_z,self.coverage_idx[0]], data_to_plot[mask_z,self.coverage_idx[1]], s=1, c='r')
                # data_obs_rho = obs_rho.reshape(-1, *self.observation_space.shape).detach().cpu().numpy() if obs_rho is not None else None
                # self.ax.scatter(data_obs_rho[:,self.coverage_idx[0]], data_obs_rho[:,self.coverage_idx[1]], s=1, c='red',alpha=0.1)  if obs_rho is not None else None
                # data_obs = obs.reshape(-1, *self.observation_space.shape).detach().cpu().numpy() if obs is not None else None
                sc = self.ax.scatter(data_to_plot[:,self.coverage_idx[0]], data_to_plot[:,self.coverage_idx[1]], s=1, c = m_n, cmap='viridis')
                # self.ax.scatter(data_obs[:,self.coverage_idx[0]], data_obs[:,self.coverage_idx[1]], s=1, c='black',alpha=0.1)  if obs is not None else None
                # self.ax.scatter(obs_un_train[:,self.coverage_idx[0]].cpu(), obs_un_train[:,self.coverage_idx[1]].cpu(), s=1, c='b', alpha=0.5)
                if hasattr(self, 'cbar'):
                    self.cbar.update_normal(sc)
                else:
                    self.cbar = self.figure.colorbar(sc, ax=self.ax)
        # else :
        #     # self.obs_saved.append(obs.cpu())
        #     # data_to_plot = torch.cat(self.obs_saved, dim=0).squeeze(1).cpu().numpy()
        #     # self.ax.scatter(data_to_plot[:,self.coverage_idx[0]], data_to_plot[:,self.coverage_idx[1]], s=1, c='black',alpha=0.1)
        #     self.ax.scatter(obs_un[:,self.coverage_idx[0]], obs_un[:,self.coverage_idx[1]], s=1, c='black', alpha=0.5)
        elif zs is not None:
            self.ax.clear()
            for i in range(zs.shape[-1]):
                # gather data
                # data = obs_un[z_un[:,i] == 1]
                data = obs[zs[:,i] == 1].cpu().numpy()
                # self.ax.scatter(data[:,self.coverage_idx[0]], data[:,self.coverage_idx[1]], s=1, c=self.random_colors[i], alpha=0.5)
                self.ax.scatter(data[:,self.coverage_idx[0]], data[:,self.coverage_idx[1]], s=1, c=self.random_colors[i], alpha=0.5)

        elif obs_un is not None : 
            self.ax.scatter(obs_un[:,self.coverage_idx[0]], obs_un[:,self.coverage_idx[1]], s=1, c='b', alpha=0.5)

        # Bounds
        self.ax.set_xlim([self.render_settings['x_lim'][0], self.render_settings['x_lim'][1]])
        self.ax.set_ylim([self.render_settings['y_lim'][0], self.render_settings['y_lim'][1]])
        if 'Maze' in self.env_id:
            for wall in self.env.walls:
                x1, y1, x2, y2 = wall
                self.ax.plot([x1, x2], [y1, y2], color='black')
        plt.grid(True)
        # save fig env_plot
        self.figure.canvas.draw()
        image = np.frombuffer(self.figure.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(self.figure.canvas.get_width_height()[::-1] + (3,))
        self.writer_gif.append_data(image)

    def set_gif(self,  name_id = None,render_settings={} ): 
            # gif 
            if not os.path.exists(self.path_gif):
                os.makedirs(self.path_gif)
            writer_gif = imageio.get_writer(self.path_gif + name_id + '.mp4', fps=1)
            return writer_gif
    
   
    def plotly(self, obs_un, obs_un_train, classifier, device , z_un = None):
        # sample 
        idx_sample = np.random.randint(0, obs_un.shape[0], self.plotly_sample_per_episode)
        # data  to plot
        data_to_plot =torch.Tensor(obs_un.reshape(-1, *self.observation_space.shape)[idx_sample]).to(device)
        # Plotting measure 
        m_n = classifier(data_to_plot).detach().cpu().numpy().squeeze(-1).copy()
        # data to plot
        data_to_plot = data_to_plot.detach().cpu().numpy().copy()
        if self.coverage_idx.shape[0] == 2:
            # plotly_2d
            add_frame_to_figure_2d(self.figure, data_to_plot[:,self.coverage_idx[0]], data_to_plot[:,self.coverage_idx[1]], m_n, self.frame)
            # plot obs train
            # add_frame_to_figure_2d(self.figure, obs_un_train[:,self.coverage_idx[0]], obs_un_train[:,self.coverage_idx[1]], np.ones(obs_un_train.shape[0]), self.frame)
        if self.coverage_idx.shape[0] == 3:
            # plotly_3d
            add_frame_to_figure_3d(self.figure, data_to_plot[:,self.coverage_idx[0]], data_to_plot[:,self.coverage_idx[1]], data_to_plot[:,self.coverage_idx[2]], m_n, self.frame)
            # plot obs train
            # add_frame_to_figure_3d(self.figure, obs_un_train[:,self.coverage_idx[0]], obs_un_train[:,self.coverage_idx[1]], data_to_plot[:,self.coverage_idx[2]], np.ones(obs_un_train.shape[0]), self.frame)
        # update frame
        self.frame += 1
    

    def signal_handler(self, sig, frame):
        """
        Gestionnaire de signal pour intercepter SIGINT (Ctrl+C).
        Args:
        sig: Le numéro du signal.
        frame: La pile d'appels courante.
        """
        if self.coverage_idx.shape[0] == 2:
            create_html_2d(self.figure, self.xp_id)
        if self.coverage_idx.shape[0] == 3:
            create_html_3d(self.figure, self.xp_id)
        sys.exit(0)  # Sortie propre du script


    def parser_robotics(self, obs):
        return obs['observation']
    
    def parser_ram_atari(self): 
        ram = self.env.unwrapped.ale.getRAM()
        if 'Montezuma' in self.env_id : 
            return {'x' : 2*((ram[42]/255)-0.5), 'y' : 2*((ram[43]/255)-0.5), 'room' : ram[57]}
        elif 'Pitfall' in self.env_id : 
            # return {'x': 2*((ram[97]/255)-0.5), 'y': 2*((ram[105]/255)-0.5), 'room': ram[98]}
            return {'x': 2*((ram[97])/(255) - 0.5), 'y': 2*((ram[105])/(255)-0.5), 'room': ram[98]}

        else : 
            raise

    def update_coverage(self, obs, infos=None):
        # if infos is not None:
        #     coords = []
        #     for info in infos['position']:
        #         coords.append([info['x'], info['y']])
        #         self.rooms.append(info['room']) if info['room'] not in self.rooms else None
        #     coords = np.array(coords)
        # if infos is None:
        coords = obs[: , self.coverage_idx].cpu().numpy()
        coords_mat =np.floor((coords - self.transposed_limits[0])/(self.transposed_limits[1]-self.transposed_limits[0])*self.coverage_accuracy).astype(np.int32)
        # check in bounds
        coords_mat = np.clip(coords_mat, 0, self.coverage_accuracy-1)
        self.matrix_coverage[tuple([coords_mat[:, i] for i in range(coords_mat.shape[1])])] += 1
       
            
                
            
    def get_coverage(self):
        return np.sum(self.matrix_coverage/(self.matrix_coverage+1e-6))/reduce(mul, self.matrix_coverage.shape)*100.0

    def shanon_entropy(self) : 
        probabilities = self.matrix_coverage/self.matrix_coverage.sum()
        entropy = (-probabilities*np.log(probabilities+1e-1)).sum()
        return entropy
    
    def get_rooms(self):
        return len(set(self.rooms))
    
    @property
    def unwrapped(self):
        return self.env.unwrapped
    
if __name__ == '__main__':
    import envs
    # Maze
    # env = Wenv('Maze', kwargs={'name': 'Easy'})
    # mujoco 
    env = Wenv('HalfCheetah-v3')
    obs, i = env.reset(seed=0)
    print(obs, i)
    print('obs shape', obs.shape)

