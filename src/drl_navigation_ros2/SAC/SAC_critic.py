import torch
from torch import nn

import SAC.SAC_utils as utils


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(self, obs_dim, action_dim, hidden_layers):
        super().__init__()

        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_layers, 1)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_layers, 1)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs["q1"] = q1
        self.outputs["q2"] = q2

        return q1, q2

    def log(self, writer, step):
        for k, v in self.outputs.items():
            writer.add_histogram(f"train_critic/{k}_hist", v, step)
