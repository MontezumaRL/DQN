from collections import deque, namedtuple
import random
import numpy as np
import torch

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity, device=None):
        """
        Initialise le buffer de replay
        Args:
            capacity: Taille maximale du buffer
            device: Device pour les tenseurs (cuda ou cpu)
        """
        self.memory = deque(maxlen=capacity)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def push(self, state, action, reward, next_state, done):
        """
        Ajoute une expérience au buffer
        Convertit les données en numpy arrays avant stockage
        """
        # Conversion des tenseurs en numpy si nécessaire
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu().numpy()
        if isinstance(done, torch.Tensor):
            done = done.cpu().numpy()

        # Conversion en numpy arrays si ce ne sont pas déjà des arrays
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = np.array(action, dtype=np.int64)
        reward = np.array(reward, dtype=np.float32)
        done = np.array(done, dtype=np.float32)

        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Échantillonne un batch d'expériences et les retourne sous forme de tenseurs
        sur le device approprié
        """
        experiences = random.sample(self.memory, batch_size)

        # Empilage des arrays numpy
        states = np.stack([exp.state for exp in experiences])
        actions = np.stack([exp.action for exp in experiences])
        rewards = np.stack([exp.reward for exp in experiences])
        next_states = np.stack([exp.next_state for exp in experiences])
        dones = np.stack([exp.done for exp in experiences])

        # Conversion en tenseurs PyTorch et déplacement vers le device approprié
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        """Vide le buffer"""
        self.memory.clear()

    def get_device(self):
        """Retourne le device utilisé par le buffer"""
        return self.device

    def set_device(self, device):
        """Change le device du buffer"""
        self.device = device