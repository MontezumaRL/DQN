# fichier dqn/src/memory/replay_buffer.py:
from collections import deque, namedtuple
import random
import numpy as np
import torch

# Définition de la structure pour stocker les expériences
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """Mémoire de replay pour stocker les expériences de l'agent"""
    
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def push(self, state, action, reward, next_state, done):
        """Ajoute une expérience à la mémoire"""
        # Conversion en numpy arrays si ce ne sont pas déjà des arrays
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state)
            
        self.memory.append(Experience(state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """Échantillonne un batch d'expériences aléatoirement et retourne des tensors PyTorch"""
        experiences = random.sample(self.memory, batch_size)
        
        # Conversion en numpy arrays
        states = np.array([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences])
        
        # Conversion en tensors PyTorch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Retourne la taille actuelle du buffer"""
        return len(self.memory)