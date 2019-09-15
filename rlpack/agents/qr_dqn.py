import numpy as np
import torch
import torch.optim as optim

import rlpack.pytorch_util as ptu
from rlpack.agent import Agent
from rlpack.eval_util import eval_np


def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


class QRDQNAgent(Agent):
    def __init__(
            self,
            z,
            z_target,

            discount=0.99,
            gradient_clipping_norm=5,
            tau=0.005,

            optimizer_class=optim.Adam,
            z_lr=1e-3,

            **kwargs,
    ):
        super().__init__(**kwargs)

        self.discount = discount
        self.gradient_clipping_norm = gradient_clipping_norm
        self.tau = tau

        self.z = z
        self.z_target = z_target

        ptu.copy_model_params_from_to(self.z, self.z_target)

        self.z_optimizer = optimizer_class(
            self.z.parameters(), lr=z_lr,
        )

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        state = batch['observations']
        actions = batch['actions']
        next_state = batch['next_observations']

        actions = actions.view(-1).long()
        theta = self.z(state)[np.arange(len(actions)), actions]

        z_next = self.z_target(next_state).detach()
        z_next_max = z_next[np.arange(len(z_next)), z_next.mean(2).max(1)[1]]
        t_theta = rewards + self.discount * (1-terminals) * z_next_max
        
        diff = t_theta.t().unsqueeze(-1) - theta
        loss = huber(diff) * (self.tau - (diff.detach() < 0).float()).abs()
        loss = loss.mean()

        self.take_optimisation_step(
            self.z_optimizer, self.z, loss, self.gradient_clipping_norm,
        )

        if self.num_train_steps % 100 == 0:
            self.z_target.load_state_dict(self.z.state_dict())

        self.writer.add_scalar('qr_dqn/loss', loss, self.num_train_steps)

    def get_actions(self, obs_np, deterministic=False):
        actions = eval_np(self.z, obs_np)
        action = actions.mean(2).max(1)[1]
        return [int(action)]

    @property
    def networks(self):
        return [
            self.z,
            self.z_target,
        ]
