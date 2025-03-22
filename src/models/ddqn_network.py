import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQNNetwork(nn.Module):
    """Réseau DQN amélioré avec architecture Dueling et Batch Normalization"""

    def __init__(self, input_shape, n_actions, dropout_rate=0.2):
        super(DuelingDQNNetwork, self).__init__()

        # Configuration des couches convolutives avec Batch Normalization
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32, track_running_stats=False),  # Désactive les stats running
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU()
        )

        # Calcul de la taille de sortie des couches convolutives
        self.conv_out_size = self._get_conv_output(input_shape)

        # Stream de valeur (V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(self.conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

        # Stream d'avantage (A(s,a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, n_actions)
        )

        # Initialisation des poids
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialisation des poids avec Kaiming normal"""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def _get_conv_output(self, shape):
        """Calcule la dimension de sortie des couches convolutives"""
        with torch.no_grad():
            # Utilise un batch de taille 2 pour éviter les problèmes avec BatchNorm
            o = self.conv(torch.zeros(2, *shape))
            return int(np.prod(o.shape[1:]))

    def forward(self, x):
        """
        Forward pass du réseau
        Args:
            x: Tensor d'entrée de forme (batch_size, channels, height, width)
        Returns:
            Tensor des Q-values pour chaque action
        """
        # Gestion du cas où x n'a pas de dimension de batch
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        # S'assurer que le batch size est au moins 2 pour BatchNorm
        if x.size(0) == 1:
            x = x.repeat(2, 1, 1, 1)
            single_input = True
        else:
            single_input = False

        # Passage à travers les couches convolutives
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)

        # Calcul de la valeur et des avantages
        value = self.value_stream(conv_out)
        advantage = self.advantage_stream(conv_out)

        # Combinaison de V(s) et A(s,a) pour Q(s,a)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        # Si l'entrée était un seul exemple, ne retourner que le premier résultat
        if single_input:
            q_values = q_values[0].unsqueeze(0)

        return q_values

    def save(self, path):
        """Sauvegarde le modèle"""
        torch.save({
            'state_dict': self.state_dict(),
            'conv_out_size': self.conv_out_size
        }, path)

    def load(self, path):
        """Charge le modèle"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])
        self.conv_out_size = checkpoint['conv_out_size']