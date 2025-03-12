# src/agents/dqn.py

import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from ..memory.replay_buffer import PrioritizedReplayBuffer, NStepReplayBuffer
from ..models.networks import DuelingDQNNetwork, NoveltyNetwork


class DQNAgent:
    """Agent d'apprentissage par renforcement utilisant DQN avec améliorations"""

    def __init__(self, input_shape, n_actions, device, config):
        self.device = device
        self.config = config
        self.n_actions = n_actions
        self.epsilon = config.EPSILON_START

        # Initialisation des réseaux
        self.policy_net = DuelingDQNNetwork(input_shape, n_actions).to(device)
        self.target_net = DuelingDQNNetwork(input_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Réseau de nouveauté pour l'exploration intrinsèque
        self.novelty_net = NoveltyNetwork(input_shape).to(device)
        self.novelty_optimizer = optim.Adam(self.novelty_net.parameters(), lr=config.NOVELTY_LR)

        # Initialisation de l'optimiseur
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)

        # Initialisation du buffer de replay
        self.memory = NStepReplayBuffer(
            config.MEMORY_SIZE,
            n_steps=config.N_STEP_RETURNS,
            gamma=config.GAMMA,
            alpha=config.PER_ALPHA,
            beta_start=config.PER_BETA_START,
            beta_frames=config.PER_BETA_FRAMES
        )

        # Buffer pour stocker les embeddings récents
        self.embedding_buffer = deque(maxlen=config.EMBEDDING_BUFFER_SIZE)

        # Facteur de scaling pour la récompense intrinsèque
        self.intrinsic_reward_scale = config.INTRINSIC_REWARD_SCALE

        # Compteur d'étapes
        self.total_steps = 0

        # Buffer pour l'entraînement du réseau de nouveauté
        self.novelty_buffer = deque(maxlen=1000)
        self.novelty_update_freq = 10  # Fréquence de mise à jour du réseau de nouveauté

    def compute_intrinsic_reward(self, state):
        """Calcule une récompense intrinsèque basée sur la nouveauté de l'état"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            embedding = self.novelty_net(state_tensor).cpu().numpy()[0]

        if len(self.embedding_buffer) == 0:
            self.embedding_buffer.append(embedding)
            return self.intrinsic_reward_scale  # Récompense maximale pour le premier état

        # Calculer la distance minimale aux embeddings précédents
        distances = [np.linalg.norm(embedding - e) for e in self.embedding_buffer]
        min_distance = min(distances)

        # Ajouter l'embedding au buffer
        self.embedding_buffer.append(embedding)

        # Stocker l'état pour l'entraînement du réseau de nouveauté
        self.novelty_buffer.append(state)

        # Entraîner le réseau de nouveauté périodiquement
        if len(self.novelty_buffer) >= 64 and self.total_steps % self.novelty_update_freq == 0:
            self._train_novelty_network()

        # La récompense intrinsèque est proportionnelle à la distance minimale
        return min_distance * self.intrinsic_reward_scale

    def _train_novelty_network(self):
        """Entraîne le réseau de nouveauté pour mieux distinguer les états"""
        if len(self.novelty_buffer) < 64:
            return

        # Échantillonner des paires d'états
        indices = np.random.choice(len(self.novelty_buffer), size=64, replace=False)
        states = [self.novelty_buffer[i] for i in indices]

        # Convertir en tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)

        # Calculer les embeddings
        embeddings = self.novelty_net(states_tensor)

        # Calculer la matrice de similarité cosinus
        similarity = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)

        # Créer les labels: 1 pour la diagonale (même état), 0 ailleurs
        labels = torch.eye(64).to(self.device)

        # Calculer la perte (contrastive loss)
        loss = F.binary_cross_entropy_with_logits(similarity, labels)

        # Optimisation
        self.novelty_optimizer.zero_grad()
        loss.backward()
        self.novelty_optimizer.step()

    def select_action(self, state):
        """Sélectionne une action selon la politique epsilon-greedy"""
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.policy_net(state_tensor).max(1)[1].item()
        else:
            return random.randint(0, self.n_actions - 1)

    def store_transition(self, state, action, reward, next_state, done):
        """Stocke une transition avec récompense intrinsèque"""
        intrinsic_reward = self.compute_intrinsic_reward(state)
        total_reward = reward + intrinsic_reward

        self.memory.push(state, action, total_reward, next_state, done)
        self.total_steps += 1

        return intrinsic_reward

    def update_epsilon(self):
        """Met à jour la valeur d'epsilon"""
        self.epsilon = max(self.config.EPSILON_END,
                           self.epsilon * self.config.EPSILON_DECAY)

    def learn(self):
        """Effectue une étape d'apprentissage avec Double DQN et Prioritized Experience Replay"""
        if len(self.memory) < self.config.BATCH_SIZE:
            return

        # Échantillonnage d'un batch d'expériences avec priorités
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.config.BATCH_SIZE)

        # Conversion en tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Calcul des Q-values actuelles
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: sélection des actions avec policy_net, évaluation avec target_net
        with torch.no_grad():
            # Sélection des actions avec policy_net
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            # Évaluation avec target_net
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q_values = rewards + self.config.GAMMA * next_q_values * (1 - dones)

        # Calcul de l'erreur TD
        td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()

        # Mise à jour des priorités
        new_priorities = td_errors + 1e-6  # Éviter les priorités nulles
        self.memory.update_priorities(indices, new_priorities)

        # Calcul de la perte avec importance sampling weights
        loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')).mean()

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

    def save(self, path, episode, metrics):
        """Sauvegarde le modèle et les métriques"""
        torch.save({
            'episode': episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'novelty_net_state_dict': self.novelty_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'novelty_optimizer_state_dict': self.novelty_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'metrics': metrics,
        }, path)

    def load(self, path):
        """Charge un modèle sauvegardé"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])

        # Charger le réseau de nouveauté si présent
        if 'novelty_net_state_dict' in checkpoint:
            self.novelty_net.load_state_dict(checkpoint['novelty_net_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Charger l'optimiseur de nouveauté si présent
        if 'novelty_optimizer_state_dict' in checkpoint:
            self.novelty_optimizer.load_state_dict(checkpoint['novelty_optimizer_state_dict'])

        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']

        return checkpoint['episode'], checkpoint.get('metrics', {})