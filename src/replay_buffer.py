# fichier: dqn/src/replay_buffer.py
from collections import deque
import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        """Ajoute une transition au buffer."""
        # Convertit les états en numpy arrays si ce n'est pas déjà fait
        state = np.array(state)
        next_state = np.array(next_state)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Échantillonne un batch de transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convertit les éléments du batch en Tensors PyTorch sur le bon device
        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        actions = torch.from_numpy(np.array(actions, dtype=np.int64)).unsqueeze(1).to(self.device) # Action: LongTensor, shape [batch, 1]
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).unsqueeze(1).to(self.device) # Reward: FloatTensor, shape [batch, 1]
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        dones = torch.from_numpy(np.array(dones, dtype=np.float32)).unsqueeze(1).to(self.device) # Done: FloatTensor (0.0 ou 1.0), shape [batch, 1]

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Retourne la taille actuelle du buffer."""
        return len(self.buffer)