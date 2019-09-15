import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import rlpack.pytorch_util as ptu
from rlpack.agent import Agent
from rlpack.eval_util import eval_np


class SACAgent(Agent):
    def __init__(self,
            critic_1,
            critic_2,
            critic_1_target,
            critic_2_target,
            actor,

            discount=0.99,
            gradient_clipping_norm=5,
            tau=0.005,

            optimizer_class=optim.Adam,
            critic_lr=1e-3,
            actor_lr=1e-3,

            use_automatic_entropy_tuning=True,
            entropy_term_weight=None,

            **kwargs,
    ):
        super().__init__(**kwargs)

        self.discount = discount
        self.gradient_clipping_norm = gradient_clipping_norm
        self.tau = tau
        
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.critic_1_target = critic_1_target
        self.critic_2_target = critic_2_target
        self.actor = actor

        ptu.copy_model_params_from_to(self.critic_1, self.critic_1_target)
        ptu.copy_model_params_from_to(self.critic_2, self.critic_2_target)

        self.critic_1_optimizer = optimizer_class(
            self.critic_1.parameters(), lr=critic_lr,
        )
        self.critic_2_optimizer = optimizer_class(
            self.critic_2.parameters(), lr=critic_lr,
        )
        self.actor_optimizer = optimizer_class(
            self.actor.parameters(), lr=actor_lr,
        )

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            self.target_entropy = -np.log((1.0 / self.action_size)) * 0.98
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha], lr=actor_lr,
            )
        else:
            self.alpha = entropy_term_weight

    @property
    def name(self):
        return 'SAC'

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        state = batch['observations']
        actions = batch['actions']
        next_state = batch['next_observations']

        qf1_loss, qf2_loss = self.calculate_critic_losses(
            state, actions, rewards, next_state, terminals,
        )
        policy_loss, log_pi = self.calculate_actor_loss(state)
        if self.use_automatic_entropy_tuning:
            alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
        else:
            alpha_loss = None
        self.update_all_parameters(qf1_loss, qf2_loss, policy_loss, alpha_loss)
        
        """
        Save some statistics for eval
        """
        self.writer.add_scalar('sac/train/qf1_loss', np.mean(ptu.get_numpy(qf1_loss)), self.num_train_steps)
        self.writer.add_scalar('sac/train/qf2_loss', np.mean(ptu.get_numpy(qf2_loss)), self.num_train_steps)
        self.writer.add_scalar('sac/train/policy_loss', np.mean(ptu.get_numpy(policy_loss)), self.num_train_steps)
        if self.use_automatic_entropy_tuning:
            self.writer.add_scalar('sac/train/alpha', self.alpha.item(), self.num_train_steps)
            self.writer.add_scalar('sac/train/alpha_loss', alpha_loss.item(), self.num_train_steps)

    def calculate_critic_losses(self, state, actions, rewards, next_state, terminals):
        """Calculates the losses for the two critics.
        """
        with torch.no_grad():
            next_state_action, _, log_action_probabilities = self.actor(next_state)
            next_state_log_pi = log_action_probabilities.gather(
                1, next_state_action.unsqueeze(-1).long())
            qf1_next_target = self.critic_1_target(next_state).gather(
                1, next_state_action.unsqueeze(-1).long())
            qf2_next_target = self.critic_2_target(next_state).gather(
                1, next_state_action.unsqueeze(-1).long())
            min_qf_next_target = torch.min(
                qf1_next_target, qf2_next_target,
            ) - self.alpha * next_state_log_pi
            next_q_value = rewards + (1.0 - terminals) * self.discount * (min_qf_next_target)
            self.critic_1_target(next_state).gather(
                1, next_state_action.unsqueeze(-1).long())
        
        qf1 = self.critic_1(state).gather(1, actions.long())
        qf2 = self.critic_2(state).gather(1, actions.long())
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss
    
    def calculate_actor_loss(self, state):
        """Calculates the loss for the actor. This loss includes the
        additional entropy term.
        """
        action, action_probabilities, log_action_probabilities = self.actor(state)
        qf1_pi = self.critic_1(state)
        qf2_pi = self.critic_2(state)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = self.alpha * log_action_probabilities - min_qf_pi
        policy_loss = action_probabilities * inside_term
        policy_loss = policy_loss.mean()
        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
        return policy_loss, log_action_probabilities

    def calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only
        relevant if self.automatic_entropy_tuning is True.
        """
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def update_all_parameters(self, critic_1_loss, critic_2_loss, actor_loss, alpha_loss):
        """Updates the parameters for the actor, both critics and (if specified) the
        temperature parameter.
        """
        self.take_optimisation_step(
            self.critic_1_optimizer, self.critic_1, critic_1_loss, self.gradient_clipping_norm,
        )
        self.take_optimisation_step(
            self.critic_2_optimizer, self.critic_2, critic_2_loss, self.gradient_clipping_norm,
        )
        self.take_optimisation_step(
            self.actor_optimizer, self.actor, actor_loss, self.gradient_clipping_norm,
        )
        ptu.soft_update_from_to(self.critic_1, self.critic_1_target, self.tau)
        ptu.soft_update_from_to(self.critic_2, self.critic_2_target, self.tau)
        if alpha_loss is not None:
            self.take_optimisation_step(self.alpha_optimizer, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()
    
    def get_actions(self, obs_np, deterministic=False):
        # Get the actions array from the returned tuple
        return eval_np(self.actor, obs_np, deterministic=deterministic)[0]

    @property
    def networks(self):
        return [
            self.actor,
            self.critic_1,
            self.critic_2,
            self.critic_1_target,
            self.critic_2_target,
        ]
    
    @property
    def params(self):
        params = {
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'critic_1_target': self.critic_1_target.state_dict(),
            'critic_2_target': self.critic_2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_1_optimizer': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer': self.critic_2_optimizer.state_dict(),
        }

        if self.use_automatic_entropy_tuning:
            params['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        
        return params
