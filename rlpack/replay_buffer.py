import abc
from collections import OrderedDict
import numpy as np


class ReplayBuffer(object, metaclass=abc.ABCMeta):
    """
    A class used to save and replay data.
    """

    def __init__(
            self,
            max_replay_buffer_size,
            env,
    ):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # self.observation_dim = self.observation_space.n
        self.observation_dim = self.observation_space.low.size
        self.action_dim = self.action_space.n
        self.max_replay_buffer_size = max_replay_buffer_size
        self.observations = np.zeros((max_replay_buffer_size, self.observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self.next_obs = np.zeros((max_replay_buffer_size, self.observation_dim))
        self.actions = np.zeros((max_replay_buffer_size, 1))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self.rewards = np.zeros((max_replay_buffer_size, 1))
        # self.terminals[i] = a terminal was received at time i
        self.terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')

        self.top = 0
        self.size = 0

    def add_sample(
        self, observation, action, reward, terminal,
        next_observation, **kwargs
    ):
        # Convert single action into a one-hot action vector
        # new_action = np.zeros(self.action_dim)
        # new_action[action] = 1
        
        self.observations[self.top] = observation
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminals[self.top] = terminal
        self.next_obs[self.top] = next_observation

        self.advance()
    
    def advance(self):
        self.top = (self.top + 1) % self.max_replay_buffer_size
        if self.size < self.max_replay_buffer_size:
            self.size += 1
    
    def random_batch(self, batch_size):
        indices = np.random.randint(0, self.size, batch_size)
        batch = dict(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            terminals=self.terminals[indices],
            next_observations=self.next_obs[indices],
        )
        return batch

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)
