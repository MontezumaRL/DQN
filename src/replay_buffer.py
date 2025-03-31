# fichier: dqn/src/replay_buffer.py
from collections import deque
import random
import numpy as np
import torch

class ReplayBuffer:
    """
    Buffer de replay standard pour stocker et échantillonner les transitions (s, a, r, s', done).
    Utilise un échantillonnage uniforme.
    """
    def __init__(self, capacity, device):
        """
        Initialise le Replay Buffer.

        Args:
            capacity (int): Taille maximale du buffer. Les transitions les plus anciennes sont retirées.
            device (torch.device): Le device (CPU ou CUDA) sur lequel les tenseurs échantillonnés doivent être placés.
        """
        # Utilise une deque pour une suppression efficace des éléments anciens lorsque la capacité est atteinte
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        """
        Ajoute une transition (expérience) au buffer.

        Args:
            state (np.ndarray): L'état observé.
            action (int): L'action entreprise.
            reward (float): La récompense reçue.
            next_state (np.ndarray): L'état suivant observé.
            done (bool): Indique si l'épisode s'est terminé après cette transition.
        """
        # S'assurer que les états sont stockés comme des numpy arrays (robustesse)
        state = np.array(state, dtype=np.float32) # Utilise float32 pour économiser mémoire vs float64
        next_state = np.array(next_state, dtype=np.float32)
        # Le type de 'done' sera converti en float32 lors de l'échantillonnage

        # Ajouter la transition complète à la deque
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Échantillonne aléatoirement un batch de transitions depuis le buffer.

        Args:
            batch_size (int): Le nombre de transitions à échantillonner.

        Returns:
            tuple: Un tuple contenant les tenseurs PyTorch pour les états, actions, récompenses,
                   prochains états et flags 'done', tous placés sur le device spécifié.
                   (states, actions, rewards, next_states, dones)
                   Shapes:
                   - states, next_states: (batch_size, C, H, W)
                   - actions, rewards, dones: (batch_size, 1)
        """
        # Vérification implicite: random.sample lèvera une erreur si batch_size > len(buffer)
        # Cette vérification est généralement faite dans l'agent avant d'appeler sample().
        batch = random.sample(self.buffer, batch_size)

        # Regrouper les éléments de même type de toutes les transitions du batch
        # states deviendra un tuple de N arrays (state), actions un tuple de N ints, etc.
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convertir les listes/tuples en un unique array numpy puis en tenseur PyTorch
        # et les déplacer vers le bon device.

        # États: Convertir en array numpy float32, puis en tenseur
        states_np = np.array(states, dtype=np.float32)
        states_tensor = torch.from_numpy(states_np).to(self.device)

        # Actions: Convertir en array numpy int64 (Long), ajouter une dimension, puis en tenseur
        actions_np = np.array(actions, dtype=np.int64)
        actions_tensor = torch.from_numpy(actions_np).unsqueeze(1).to(self.device)

        # Récompenses: Convertir en array numpy float32, ajouter une dimension, puis en tenseur
        rewards_np = np.array(rewards, dtype=np.float32)
        rewards_tensor = torch.from_numpy(rewards_np).unsqueeze(1).to(self.device)

        # Prochains états: Convertir en array numpy float32, puis en tenseur
        next_states_np = np.array(next_states, dtype=np.float32)
        next_states_tensor = torch.from_numpy(next_states_np).to(self.device)

        # Dones: Convertir en array numpy float32 (pour calculs comme 1-dones), ajouter une dimension, puis en tenseur
        dones_np = np.array(dones, dtype=np.float32)
        dones_tensor = torch.from_numpy(dones_np).unsqueeze(1).to(self.device)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def __len__(self):
        """Retourne le nombre actuel de transitions stockées dans le buffer."""
        return len(self.buffer)