# fichier: dqn/src/networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN_CNN(nn.Module):
    """
    Architecture CNN standard pour DQN sur Atari (basée sur Nature DQN).
    Prend en entrée un stack de frames prétraitées (ex: 4x42x42)
    et sort les Q-values pour chaque action.
    """
    def __init__(self, input_shape, n_actions):
        """
        Args:
            input_shape (tuple): Forme de l'état d'entrée (C, H, W).
            n_actions (int): Nombre d'actions possibles.
        """
        super(DQN_CNN, self).__init__()
        # Input shape: (C, H, W), ex: (4, 42, 42)
        self.conv = nn.Sequential(
            # Couche 1: C entrées, 32 sorties, kernel 8x8, stride 4
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Couche 2: 32 entrées, 64 sorties, kernel 4x4, stride 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Couche 3: 64 entrées, 64 sorties, kernel 3x3, stride 1
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calculer la taille de la sortie aplatie des couches convolutives
        conv_out_size = self._get_conv_out(input_shape)

        # Couches Fully Connected (Linéaires)
        self.fc = nn.Sequential(
            # Couche linéaire 1: sortie conv aplatie -> 512 neurones
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            # Couche linéaire 2 (sortie): 512 neurones -> n_actions (Q-values)
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        """Calcule la taille de sortie des couches convolutives pour une forme d'entrée donnée."""
        # Crée un tenseur dummy de la bonne forme
        o = self.conv(torch.zeros(1, *shape))
        # Retourne le nombre total d'éléments dans le tenseur de sortie aplati
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        Passe avant du réseau.

        Args:
            x (torch.Tensor): Tenseur d'entrée (batch_size, C, H, W).

        Returns:
            torch.Tensor: Q-values pour chaque action (batch_size, n_actions).
        """
        # Passe à travers les couches convolutives
        conv_out = self.conv(x)
        # Aplatit la sortie des convolutions pour les couches FC
        # view(x.size()[0], -1) réorganise en (batch_size, features_aplati)
        flattened_conv_out = conv_out.view(x.size()[0], -1)
        # Passe à travers les couches fully connected
        q_values = self.fc(flattened_conv_out)
        return q_values

class RNDNetwork(nn.Module):
    """
    Architecture pour les réseaux RND (Target et Predictor).
    Utilise la même base CNN que DQN pour extraire les features,
    puis une couche linéaire pour produire un embedding.
    """
    def __init__(self, input_shape, output_dim=512):
        """
        Args:
            input_shape (tuple): Forme de l'état d'entrée normalisé (C, H, W).
            output_dim (int): Dimension de l'embedding de sortie RND.
        """
        super(RNDNetwork, self).__init__()
        # Réutilise la même architecture convolutive que DQN
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        # Couche linéaire finale pour l'embedding RND
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, output_dim)
            # Pas d'activation ici généralement pour l'embedding RND final
        )
        # Initialisation spécifique (par exemple orthogonale) pourrait être testée ici,
        # mais l'initialisation par défaut est souvent suffisante.

    def _get_conv_out(self, shape):
        """Helper pour calculer la taille de sortie des convolutions."""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        Passe avant du réseau RND.

        Args:
            x (torch.Tensor): Tenseur d'entrée normalisé (batch_size, C, H, W).

        Returns:
            torch.Tensor: Embedding RND (batch_size, output_dim).
        """
        # Extraction des features par les convolutions
        conv_out = self.conv(x)
        # Aplatir
        flattened_conv_out = conv_out.view(x.size()[0], -1)
        # Projection linéaire pour obtenir l'embedding
        embedding = self.fc(flattened_conv_out)
        return embedding