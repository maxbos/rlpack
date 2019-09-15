import abc
from collections import OrderedDict

from typing import Iterable
from torch import nn as nn

from rlpack.eval_util import torch_ify
from rlpack.pytorch_util import np_to_pytorch_batch, from_numpy


class Agent(object, metaclass=abc.ABCMeta):
    def __init__(self, env):
        self.env = env
        self.num_train_steps = 0
        self.action_size = self.env.action_space.n

    def set_writer(self, writer):
        self.writer = writer

    def train(self, np_batch):
        self.num_train_steps += 1
        batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch(batch)
    
    @abc.abstractmethod
    def train_from_torch(self, batch):
        pass
    
    def take_optimisation_step(
        self, optimizer, network, loss, clipping_norm=None, retain_graph=False
    ):
        """Takes an optimisation step by calculating gradients given the loss and
        then updating the parameters.
        """
        if not isinstance(network, list): network = [network]
        # reset gradients to 0
        optimizer.zero_grad()
        # this calculates the gradients
        loss.backward(retain_graph=retain_graph)
        if clipping_norm is not None:
            for net in network:
                # clip gradients to help stabilise training
                nn.utils.clip_grad_norm_(net.parameters(), clipping_norm)
        # this applies the gradients
        optimizer.step()

    def get_action(self, obs_np, deterministic=False):
        """
        :param observation:
        :return: action, debug_dictionary
        """
        # Indexation by None turns the single sample into a batch matrix
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0]
    
    @abc.abstractmethod
    def get_actions(self, obs_np, deterministic=False):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[nn.Module]:
        pass

    @property
    @abc.abstractmethod
    def params(self) -> dict:
        pass
    
    def eval(self, enable=False):
        for net in self.networks:
            net.eval() if enable else net.train()
