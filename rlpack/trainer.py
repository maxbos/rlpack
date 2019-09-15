import abc
import os
import numpy as np
from collections import OrderedDict
from datetime import datetime

import torch
from tensorboardX import SummaryWriter

import rlpack.eval_util as eval_util
from rlpack.replay_buffer import ReplayBuffer


class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            agent,
            env,
            replay_buffer: ReplayBuffer,
            batch_size,
            num_episodes,
            num_episodes_before_evaluation,
            num_steps_before_training_starts,
            num_steps_before_training_repeats=1,
            num_train_calls_per_step=1,
            num_trial_episodes_per_evaluation=10,
            num_steps_before_on_policy_action=None,
            epsilon=None,
    ):
        self.writer = SummaryWriter()

        self.agent = agent
        self.env = env
        self.replay_buffer = replay_buffer

        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.num_episodes_before_evaluation = num_episodes_before_evaluation
        self.num_trial_episodes_per_evaluation = num_trial_episodes_per_evaluation
        self.num_steps_before_training_starts = num_steps_before_training_starts
        self.num_steps_before_training_repeats = num_steps_before_training_repeats
        self.num_train_calls_per_step = num_train_calls_per_step

        self.num_steps_before_on_policy_action = num_steps_before_on_policy_action
        self.epsilon = epsilon

        self.current_episode_number = 0
        self.current_step_number = 0

        self.agent.set_writer(self.writer)
        self.started_at = str(datetime.now()).replace(' ', '_')

    def train(self):
        """Train the agent for a given number of episodes.
        """
        self.current_episode_number = 0
        self.current_step_number = 0

        for episode_i in range(1, self.num_episodes+1):
            print('Starting training for episode {}...'.format(episode_i))
            self.current_episode_number = episode_i
            state = self.env.reset()
            returns = 0
            done = False

            while not done:
                next_state, reward, done, _ = self.step(state)
                # Every step is stored to the replay buffer. Only consider training
                # if the environment has been explored for a minimum number of steps.
                if self.replay_buffer.size >= self.num_steps_before_training_starts:
                    # This regulates the training frequency. By setting the number of steps
                    # before training repeats one can specify that training should only
                    # be performed after a certain number of steps.
                    if self.current_step_number % self.num_steps_before_training_repeats == 0:
                        # One can set to perform multiple training calls per episode step.
                        for _ in range(self.num_train_calls_per_step):
                            batch = self.replay_buffer.random_batch(self.batch_size)
                            self.agent.train(batch)
                
                state = next_state
                returns += reward
            
            if episode_i % self.num_episodes_before_evaluation == 0:
                print('Starting evaluation...')
                self.save_params(episode_i)
                self.test()
        
        self.env.close()
        self.save_params(episode_i)
        self.test()

    def step(self, state, is_test=False):
        """Select an action (randomly or from policy), perform the action on the
        environment, and add the newly observed information to the replay buffer.
        """
        action = self.select_action(state, is_test)
        next_state, reward, done, _ = self.env.step(action)
        self.replay_buffer.add_sample(state, action, reward, done, next_state)
        if not is_test:
            self.current_step_number += 1
        return next_state, reward, done, action

    def select_action(self, state, is_test=False):
        """Return a random action depending on the total number of steps if the steps
        before policy action threshold is set or on chance if the exploration
        vs. exploitation epsilon fraction is set.
        """
        if (
            self.num_steps_before_on_policy_action and
            self.current_step_number < self.num_steps_before_on_policy_action
        ) or (
            self.epsilon and self.epsilon > np.random.random()
        ):
            return self.env.action_space.sample()
        action = self.agent.get_action(state, deterministic=is_test)
        return int(action)
    
    def test(self):
        """Runs an evaluation test of the current agent by running a given
        number of trial episodes and each step performing the on policy action
        with the highest probability (i.e. deterministic).
        """
        episode_returns = list()
        episode_actions = list()
        episode_rewards = list()
        with torch.no_grad():
            self.agent.eval(enable=True)
            for episode_i in range(self.num_trial_episodes_per_evaluation):
                state = self.env.reset()
                done = False
                returns = 0
                while not done:
                    next_state, reward, done, action = self.step(state, is_test=True)
                    state = next_state
                    returns += reward
                    episode_actions.append(action)
                    episode_rewards.append(reward)
                episode_returns.append(returns)
            self.agent.eval(enable=False)
        # Write results of the evaluation to log
        episode_returns = np.array(episode_returns)
        episode_actions = np.array(episode_actions)
        episode_rewards = np.array(episode_rewards)
        self.writer.add_scalar('test/returns_min', episode_returns.min(), self.current_episode_number)
        self.writer.add_scalar('test/returns_avg', episode_returns.mean(), self.current_episode_number)
        self.writer.add_scalar('test/returns_max', episode_returns.max(), self.current_episode_number)
        self.writer.add_scalar('test/returns_std', episode_returns.std(), self.current_episode_number)
        self.writer.add_scalar('test/actions_avg', episode_actions.mean(), self.current_episode_number)
        self.writer.add_scalar('test/rewards_min', episode_rewards.min(), self.current_episode_number)
        self.writer.add_scalar('test/rewards_avg', episode_rewards.mean(), self.current_episode_number)
        self.writer.add_scalar('test/rewards_max', episode_rewards.max(), self.current_episode_number)
        self.writer.add_scalar('test/rewards_std', episode_rewards.std(), self.current_episode_number)

    def save_params(self, episode_i):
        if not os.path.exists('./save'):
            os.mkdir('./save')

        save_name = '{}_{}_ep_{}.pt'.format(
            self.agent.name, self.started_at, str(episode_i))
        path = os.path.join('./save', save_name)
        torch.save(self.agent.params, path)

        print('[INFO] Saved the model and optimizer to', path)

    def to(self, device):
        for net in self.agent.networks:
            net.to(device)
