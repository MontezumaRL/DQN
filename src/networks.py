# fichier: dqn/src/networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN_CNN(nn.Module):
    """Architecture CNN pour DQN sur Atari"""
    def __init__(self, input_shape, n_actions):
        super(DQN_CNN, self).__init__()
        # Input shape: (C, H, W) = (4, 84, 84)
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calculer la taille de sortie après les convolutions
        conv_out_size = self._get_conv_out(input_shape)

        # Couches fully connected (dueling network architecture pourrait être mieux, mais restons simple pour commencer)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        # Fonction helper pour calculer la taille après les convolutions
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # x shape: (batch_size, C, H, W)
        conv_out = self.conv(x).view(x.size()[0], -1) # Aplatir la sortie des convs
        return self.fc(conv_out)

class RNDNetwork(nn.Module):
    """Architecture pour les réseaux RND (Target et Predictor)"""
    def __init__(self, input_shape, output_dim=512):
        super(RNDNetwork, self).__init__()
        # Utilise la même base convolutive que DQN
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        # Couches fully connected pour l'embedding RND
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, output_dim),
            # nn.ReLU(), # Optionnel: ajouter une activation?
            # nn.Linear(512, output_dim) # Optionnel: une couche de plus?
        )
         # Initialisation spécifique pour RND si nécessaire (souvent pas critique)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # x shape: (batch_size, C, H, W) - Doit être l'état *normalisé*
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)