# dqn/src/models/networks.py
import numpy as np
import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """Réseau de neurones pour l'approximation de la fonction Q"""

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
        """Calcule la taille de sortie du bloc convolutif"""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """Propagation avant"""
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
