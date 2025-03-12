# src/memory/replay_buffer.py

import numpy as np
from collections import namedtuple, deque
import random

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """Mémoire de replay standard"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Ajoute une expérience à la mémoire"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Échantillonne un batch aléatoire d'expériences"""
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayBuffer:
    """Mémoire de replay avec échantillonnage prioritaire"""

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha  # Détermine l'importance de la priorité
        self.beta_start = beta_start  # Importance sampling weight
        self.beta_frames = beta_frames
        self.frame = 1  # Pour l'annealing de beta
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Ajoute une expérience avec priorité maximale"""
        max_priority = self.priorities.max() if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append(Experience(state, action, reward, next_state, done))
        else:
            self.memory[self.position] = Experience(state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Échantillonne un batch selon les priorités"""
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]

        # Calculer les probabilités d'échantillonnage
        probs = prios ** self.alpha
        probs /= probs.sum()

        # Calculer beta pour l'importance sampling
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        # Échantillonner les indices selon les probabilités
        indices = np.random.choice(len(self.memory), batch_size, p=probs)

        # Calculer les poids d'importance sampling
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        experiences = [self.memory[idx] for idx in indices]

        states = np.array([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences])

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        """Met à jour les priorités des expériences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)


class NStepReplayBuffer:
    """Buffer pour stocker les transitions avec retours à n étapes"""

    def __init__(self, capacity, n_steps=3, gamma=0.99, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.buffer = PrioritizedReplayBuffer(capacity, alpha, beta_start, beta_frames)
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_steps)

    def _get_n_step_info(self):
        """Calcule le retour à n étapes et l'état final"""
        reward, next_state, done = self.n_step_buffer[-1][-3:]

        for _, _, r, next_s, d in reversed(list(self.n_step_buffer)[:-1]):
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (next_s, d) if d else (next_state, done)

        return reward, next_state, done

    def push(self, state, action, reward, next_state, done):
        """Ajoute une transition au buffer n-step"""
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) < self.n_steps:
            return

        # Calculer le retour à n étapes
        reward, next_state, done = self._get_n_step_info()
        state, action, _, _, _ = self.n_step_buffer[0]

        # Ajouter au buffer principal
        self.buffer.push(state, action, reward, next_state, done)

    def sample(self, batch_size):
        """Échantillonne un batch du buffer principal"""
        return self.buffer.sample(batch_size)

    def update_priorities(self, indices, priorities):
        """Met à jour les priorités"""
        self.buffer.update_priorities(indices, priorities)

    def __len__(self):
        return len(self.buffer)