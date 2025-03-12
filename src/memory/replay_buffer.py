# fichier dqn/src/memory/replay_buffer.py:
from collections import deque, namedtuple
import random
import numpy as np

# Définition de la structure pour stocker les expériences
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """Mémoire de replay pour stocker les expériences de l'agent"""
    
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Ajoute une expérience à la mémoire"""
        self.memory.append(Experience(state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """Échantillonne un batch d'expériences aléatoirement"""
        experiences = random.sample(self.memory, batch_size)
        
        states = np.array([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)