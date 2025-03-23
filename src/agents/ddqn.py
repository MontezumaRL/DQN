from ..models.ddqn_network import DuelingDQNNetwork
import torch
import torch.nn.functional as F
import numpy as np
import os

class DuelingDQNAgent:
    """Agent DQN amélioré avec plusieurs optimisations"""

    def __init__(self, input_shape, n_actions, device, config):
        self.device = device
        self.config = config
        self.n_actions = n_actions

        # Réseaux principal et cible
        self.policy_net = DuelingDQNNetwork(input_shape, n_actions).to(device)
        self.target_net = DuelingDQNNetwork(input_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimiseur avec gradient clipping
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=getattr(config, 'WEIGHT_DECAY', 0)
        )

        # Paramètres d'exploration
        self.epsilon = config.EPSILON_START
        self.epsilon_end = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY

        # Compteurs et paramètres
        self.total_steps = 0
        self.training_steps = 0
        self.target_update_freq = getattr(config, 'TARGET_UPDATE_FREQ', 1000)

        # Buffer pour N-step returns
        self.n_step_buffer = []
        self.gamma = config.GAMMA
        self.n_steps = getattr(config, 'N_STEPS', 1)

    def select_action(self, state, training=True):
        """Sélectionne une action selon la politique epsilon-greedy"""
        if training and np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
            return torch.tensor([action], dtype=torch.long)  # Retourne un tenseur 1D

        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state)

            state = state.unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].cpu()  # Retourne un tenseur 1D sur CPU

    def update_epsilon(self):
        """Met à jour le taux d'exploration epsilon"""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )

    def _compute_n_step_returns(self, reward, next_state, done):
        """Calcule les returns n-step"""
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, dtype=torch.float32)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state)
        if not isinstance(done, torch.Tensor):
            done = torch.tensor(done, dtype=torch.float32)

        self.n_step_buffer.append((reward, next_state, done))

        if len(self.n_step_buffer) < self.n_steps:
            return None

        reward, next_state, done = self.n_step_buffer.pop(0)

        # Calcul du return n-step
        for i, (r, _, _) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** (i + 1)) * r

        return reward, next_state, done

    def learn(self, experiences):
        """Effectue une étape d'apprentissage"""
        states, actions, rewards, next_states, dones = experiences

        # S'assurer que les actions sont dans le bon format pour le gather
        actions = actions.view(-1, 1)  # Reshape en (batch_size, 1)
        rewards = rewards.view(-1, 1)  # Reshape en (batch_size, 1)
        dones = dones.view(-1, 1)  # Reshape en (batch_size, 1)

        # Calcul des Q-values actuelles
        current_q_values = self.policy_net(states).gather(1, actions)

        # Double DQN: sélection des actions avec le réseau principal
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards + (self.gamma ** self.n_steps) * next_q_values * (1 - dones)

        # Calcul de la perte Huber
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # Optimisation
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping si configuré
        if hasattr(self.config, 'GRAD_CLIP'):
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.GRAD_CLIP)

        self.optimizer.step()
        self.training_steps += 1

        # Mise à jour du réseau cible si nécessaire
        if self.training_steps % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        """Mise à jour douce du réseau cible"""
        tau = getattr(self.config, 'TAU', 1.0)  # Utilise TAU=1.0 par défaut (hard update)

        for target_param, policy_param in zip(
            self.target_net.parameters(),
            self.policy_net.parameters()
        ):
            target_param.data.copy_(
                tau * policy_param.data +
                (1 - tau) * target_param.data
            )

    def save(self, path, additional_info=None):
        """Sauvegarde du modèle avec des informations supplémentaires"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_dict = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'training_steps': self.training_steps
        }

        if additional_info is not None:
            save_dict.update(additional_info)

        torch.save(save_dict, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Chargement du modèle et des informations d'état"""
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return None

        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']
        self.training_steps = checkpoint['training_steps']

        print(f"Model loaded from {path}")
        return checkpoint