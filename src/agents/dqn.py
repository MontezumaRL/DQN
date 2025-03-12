import random

import torch
import torch.nn as nn
import torch.optim as optim

from ..memory.replay_buffer import ReplayBuffer
from ..models.networks import DQNNetwork


class DQNAgent:
    """Agent d'apprentissage par renforcement utilisant DQN"""

    def __init__(self, input_shape, n_actions, device, config):
        self.device = device
        self.config = config
        self.n_actions = n_actions
        self.epsilon = config.EPSILON_START

        # Initialisation des réseaux
        self.policy_net = DQNNetwork(input_shape, n_actions).to(device)
        self.target_net = DQNNetwork(input_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Initialisation de l'optimiseur
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)

        # Initialisation du buffer de replay
        self.memory = ReplayBuffer(config.MEMORY_SIZE)

        # Compteur d'étapes
        self.total_steps = 0

    def select_action(self, state):
        """Sélectionne une action selon la politique epsilon-greedy"""
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.policy_net(state_tensor).max(1)[1].item()
        else:
            return random.randint(0, self.n_actions - 1)

    def store_transition(self, state, action, reward, next_state, done):
        """Stocke une transition dans la mémoire de replay"""
        self.memory.push(state, action, reward, next_state, done)
        self.total_steps += 1

    def update_epsilon(self):
        """Met à jour la valeur d'epsilon"""
        self.epsilon = max(self.config.EPSILON_END,
                           self.epsilon * self.config.EPSILON_DECAY)

    def learn(self):
        """Effectue une étape d'apprentissage"""
        if len(self.memory) < self.config.BATCH_SIZE:
            return

        # Échantillonnage d'un batch d'expériences
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.BATCH_SIZE)

        # Conversion en tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Calcul des Q-values actuelles
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Calcul des Q-values cibles
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.config.GAMMA * next_q_values * (1 - dones)

        # Calcul de la perte
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Optimisation
        self.optimizer.zero_grad()
        loss.backward()
        # Clip du gradient pour stabilité
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Met à jour le réseau cible"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path, episode, episode_rewards):
        """Sauvegarde le modèle"""
        torch.save({
            'episode': episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'episode_rewards': episode_rewards,
        }, path)

    def load(self, path):
        """Charge un modèle sauvegardé"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']
        return checkpoint['episode'], checkpoint['episode_rewards']