import os
import sys
from src.utils.evonump import NeuralNetNumpy, NumpyLayer
import ray 
import json
import numpy as np

@ray.remote
class Individual:
	""" Deterministic Policy """
	def __init__(self,env,genome=[],std=0.001,
				smooth_tanh_mean=0.01,smooth_tanh_std=0.01,
				log_std_max = -2, log_std_min = -5):
		# super(Individual, self).__init__()
		self.env=env
		observation_shape, action_shape=env.observation_space.shape, env.action_space.shape
		### LAYERS ###
		self.nn=NeuralNetNumpy()
		self.nn.dense(NumpyLayer(input=observation_shape[0],unit=16, activation='relu',name="LR1",std_w=np.sqrt(2.0)))
		self.nn.dense(NumpyLayer(input=16,unit=16, activation='relu',name="LR2", std_w=np.sqrt(2.0)))
		self.nn.dense(NumpyLayer(input=16,unit=env.action_space.shape[0], activation='linear',name="a_mean", std_w=0.01))
		self.nn.dense(NumpyLayer(input=16,unit=env.action_space.shape[0], activation='linear',name="a_log_std", std_w=0.01))
		self.smooth_tanh_mean=smooth_tanh_mean
		self.smooth_tanh_std=smooth_tanh_std
		self.log_std_max = log_std_max
		self.log_std_min = log_std_min
		if len(genome)>0:
			self.nn.genome=genome
		else: 
			self.nn.genome=np.random.normal(0,std,self.nn.genome.shape[0])
		
	def __call__(self, x):
		x = self.nn.layers['LR1'](x.T)
		x = self.nn.layers['LR2'](x)
		a_mean=self.nn.layers['a_mean'](x)
		a_log_std = self.nn.layers['a_log_std'](x)
		return a_mean, a_log_std

	def policy(self,s):
		s=np.expand_dims(s, axis=0)
		a_mean, a_log_std=self(s)
		a_mean= np.tanh(a_mean/self.smooth_tanh_mean)
		a_log_std = np.tanh(a_log_std/self.smooth_tanh_std)
		a_log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (a_log_std + 1)
		a = a_mean + np.exp(a_log_std) * np.random.randn(*a_mean.shape)
		return a

	def act(self,s):
		s=np.expand_dims(s, axis=0)
		action=self(s)
		return action

	def genome(self):
		return self.nn.genome
	
	def set_genome(self, genome):
		self.nn.genome=genome
		
	def eval(self,gen,env):
		# set genome
		self.set_genome(gen)
		# max_steps
		max_steps=env.config['max_episode_steps']	
		# reset
		s,i=env.reset()
		# done
		d=False
		# reward
		fitness=0
		# list states 
		states=[s.reshape(1,-1).copy()]
		time = 0
		# rollout evaluation
		while not d and time<max_steps:
			# act
			# a=self.act(s)
			a=self.policy(s)
			a=a.flatten() 
			# step
			s, r, d, tr, i=env.step(a)
			# add state
			states.append(s.reshape(1,-1).copy())
			# value updated
			fitness+=r
			# bcoord
			# b=info['bcoord']
			time += 1	
		data = np.concatenate(states,axis=0)
		dones = np.zeros_like(data[:,0])
		dones[-1] = 1
		return {"genome": self.nn.genome,"fitness": fitness, 'data': data, 'dones': dones}


	def save(self,name='genome'):
		data={'genome':list(self.nn.genome)}
		with open('/checkpoints/'+name+'.json', 'w') as f:
			json.dump(data, f)

	def load(self,name='genome'):
		with open('/checkpoints/'+name+'.json') as f:
			data = json.load(f)
			self.nn.genome=np.array(data['genome'])

	def terminate(self):
		print("Self-killing")
		# ray.actor.exit_actor()
		# os._exit(0)




if __name__=="__main__":
	from envs.config_env import config
	from envs.wenv import Wenv
	import time
	env = Wenv("HalfCheetah-v3", **config["HalfCheetah-v3"])
	# Individual
	ind=Individual.remote(env)
	# check eval
	t = time.time()
	result = ray.get(ind.eval.remote(env))
	print('value:', result['value'])
	print('data shape:', result['data'].shape)
	print('Time:', time.time()-t)

