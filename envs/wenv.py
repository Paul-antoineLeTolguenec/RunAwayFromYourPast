import matplotlib.pyplot as plt
import numpy as np
import gym, torch, os , imageio
from src.utils.dash_utils_2d import initialize_figure_2d, add_frame_to_figure_2d, create_html_2d
from src.utils.dash_utils_3d import initialize_figure_3d, add_frame_to_figure_3d, create_html_3d
import sys, signal

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
        # load the environment
        self.env = gym.make(env_id, **kwargs)
        # observation space
        self.observation_space = self.env.observation_space
        # action space
        self.action_space = self.env.action_space
        # spec 
        self.spec = self.env.spec
        # metrics
        self.coverage_accuracy = coverage_accuracy
        self.matrix_coverage = np.zeros((coverage_idx.shape[0], self.coverage_accuracy))
        self.coverage_idx = coverage_idx
        self.episode_length = 0
        self.episode_reward = 0
        # figure 
        if render_bool_matplot:
            self.figure, self.ax = plt.subplots()
            self.render_settings = render_settings
            self.ax.set_xlim([render_settings['x_lim'][0], render_settings['x_lim'][1]])
            self.ax.set_ylim([render_settings['y_lim'][0], render_settings['y_lim'][1]])
            self.writer_gif = self.set_gif(xp_id)
        # plotly
        if render_bool_plotly:
            self.frame = 0
            self.plotly_sample_per_episode = plotly_sample_per_episode
            signal.signal(signal.SIGINT, self.signal_handler)
            if self.coverage_idx.shape[0] == 2:
                self.figure = initialize_figure_2d(render_settings=render_settings)
            if self.coverage_idx.shape[0] == 3:
                self.figure = initialize_figure_3d(render_settings=render_settings)

    def set_all_seeds(self, seed):
        self.seed(seed)
        self.env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def reset(self, seed = 0):
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
        return obs, i
    

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
        return obs, reward, done, trunc, info
    
    def render(self):
        return self.env.render()
    def close(self):
        return self.env.close()
    def seed(self, seed=None):
        self.env.seed = seed
    def __str__(self):
        return self.env.__str__()
    
    def gif(self, obs_un, obs_un_train, obs, classifier, device , z_un = None): 
        with torch.no_grad():
            # clear the plot
            self.ax.clear()
            # data  to plot
            data_to_plot =torch.Tensor(obs_un.reshape(-1, *self.observation_space.shape)).to(device)
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
            self.ax.scatter(data_to_plot[:,self.coverage_idx[0]], data_to_plot[:,self.coverage_idx[1]], s=1, c = m_n, cmap='viridis')
            # plot obs train
            # self.ax.scatter(data_to_plot[mask,self.coverage_idx[0]], data_to_plot[mask,self.coverage_idx[1]], s=1, c='g')
            # self.ax.scatter(data_to_plot[mask_z,self.coverage_idx[0]], data_to_plot[mask_z,self.coverage_idx[1]], s=1, c='r')
            self.ax.scatter(obs_un_train[:,self.coverage_idx[0]].cpu(), obs_un_train[:,self.coverage_idx[1]].cpu(), s=1, c='b', alpha=0.5)
            data_obs = obs.reshape(-1, *self.observation_space.shape).detach().cpu().numpy() if obs is not None else None
            self.ax.scatter(data_obs[:,self.coverage_idx[0]], data_obs[:,self.coverage_idx[1]], s=1, c='black',alpha=0.1)  if obs is not None else None

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
            if not os.path.exists('gif'):
                os.makedirs('gif')
            writer_gif = imageio.get_writer('gif/{}.mp4'.format(name_id), fps=1)
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
            return {'x' : 2*((ram[42]/255)-0.5), 'y' : 2*((ram[43]/255)-0.5)}
        elif 'Pitfall' in self.env_id : 
            return {'x': 2*((ram[97]/255)-0.5), 'y': 2*((ram[105]/255)-0.5)}
        else : 
            raise

    def update_coverage(self, obs, info=None):
        # update coverage
        if info is None:
            self.matrix_coverage[:, i] = obs[ : , self.coverage_idx]
if __name__ == '__main__':
    import envs
    # Maze
    # env = Wenv('Maze', kwargs={'name': 'Easy'})
    # mujoco 
    env = Wenv('HalfCheetah-v3')
    obs, i = env.reset(seed=0)
    print(obs, i)
    print('obs shape', obs.shape)

