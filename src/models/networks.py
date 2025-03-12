# src/models/networks.py

import torch
import torch.nn as nn
import numpy as np


class DuelingDQNNetwork(nn.Module):
    """Réseau Dueling DQN pour l'approximation de la fonction Q"""

    def __init__(self, input_shape, n_actions):
        super(DuelingDQNNetwork, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_output(input_shape)

        # Stream pour la valeur d'état
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Stream pour les avantages des actions
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_output(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)

        value = self.value_stream(conv_out)
        advantages = self.advantage_stream(conv_out)

        # Combiner valeur et avantages pour obtenir les Q-values
        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum(A(s,a')))
        return value + (advantages - advantages.mean(dim=1, keepdim=True))


class DQNNetwork(nn.Module):
    """Réseau DQN standard pour l'approximation de la fonction Q"""

    def __init__(self, input_shape, n_actions):
        super(DQNNetwork, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_output(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_output(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class NoveltyNetwork(nn.Module):
    """Réseau pour estimer la nouveauté des états"""

    def __init__(self, input_shape):
        super(NoveltyNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_output(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128)  # Embedding de dimension 128
        )

    def _get_conv_output(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

    def reset_parameters(self):
        """Initialise les poids avec une variance plus élevée pour encourager la diversité"""
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.5)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)