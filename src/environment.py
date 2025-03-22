# fichier dqn/src/environment.py
from collections import deque

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import torch
import ale_py

from .rewards import RewardSystem
from .utils import preprocess_frame


class MontezumaEnvironment:
    def __init__(self, render_mode):
        self.env = gymnasium.make("ALE/MontezumaRevenge-v5", render_mode=render_mode)
        self.ale = self.env.unwrapped.ale
        self.frame_stack = deque(maxlen=4)
        self.action_space = self.env.action_space
        self.n_actions = self.env.action_space.n
        self.lives = 0

        # Nouveau système de récompenses
        self.reward_system = RewardSystem()

        # Paramètres de l'épisode
        self.steps_in_episode = 0
        self.max_steps = 1000

    def reset(self, seed=None, options=None):
        """
        Réinitialise l'environnement et retourne l'état initial.

        Args:
            seed (int, optional): Graine pour la génération aléatoire
            options (dict, optional): Options supplémentaires pour la réinitialisation

        Returns:
            numpy.ndarray: État initial (stack de frames)
        """
        # Passer le seed à l'environnement sous-jacent
        state = self.env.reset(seed=seed, options=options)[0]
        state = preprocess_frame(state)
        # Réinitialiser le compteur de pas
        self.steps_in_episode = 0
        self.reward_system.reset()

        # Initialiser le stack de frames
        for _ in range(4):
            self.frame_stack.append(state)

        # Initialiser le nombre de vies
        _, _, _, _, info = self.env.step(0)  # Action NOOP pour obtenir info
        self.lives = info.get('lives', 0)
        self.setup_position_tp()

        return np.array(self.frame_stack)

    def step(self, action):
        next_state, base_reward, terminated, truncated, info = self.env.step(action)
        next_state = preprocess_frame(next_state)
        self.frame_stack.append(next_state)
        # Incrémenter le compteur de pas et appliquer la pénalité de temps
        self.steps_in_episode += 1

        done = terminated or truncated

        # Vérifier la perte de vie
        current_lives = info.get('lives', 0)
        life_lost = current_lives < self.lives
        self.lives = current_lives

        # On arrête l'épisode si on perd une vie
        if life_lost:
            done = True

        # Vérifier le timeout, on arrête l'épisode si on atteint le nombre maximal de pas
        timeout = self.steps_in_episode >= self.max_steps
        if timeout:
            done = True

        # Obtenir la position de l'agent
        x, y = self.get_agent_position()

        # Calculer la récompense totale
        reward = self.reward_system.calculate_total_reward(
            base_reward=base_reward,
            x=x,
            y=y,
            life_lost=life_lost,
            timeout=timeout
        )

        if reward > 50:
            done = True

        return np.array(self.frame_stack), reward, done, info

    def get_state_tensor(self, device):
        return torch.FloatTensor(np.array(self.frame_stack)).unsqueeze(0).to(device)

    def display_frame_stack(self, save_path="frame_stack.png"):
        """Affiche les 4 frames de la stack actuelle ou les sauvegarde dans un fichier"""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for i, frame in enumerate(self.frame_stack):
            axes[i].imshow(frame, cmap='gray')
            axes[i].set_title(f'Frame {i + 1}')
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)  # Ferme la figure pour libérer la mémoire
        print(f"Frame stack sauvegardée dans {save_path}")

    def get_agent_position(self):
        """Récupère la position de l'agent dans la RAM ALE"""
        ram = self.ale.getRAM()

        return int(ram[42]), int(ram[43])

    def set_agent_position(self, x, y):
        """Déplace l'agent à la position (x, y) dans la RAM ALE"""
        self.ale.setRAM(42, x)
        self.ale.setRAM(43, y)

    def close(self):
        self.env.close()

    def setup_position_human(self):
        self.env.step(14)
        self.env.step(3)
        self.env.step(3)
        self.env.step(3)
        self.env.step(3)
        self.env.step(3)
        self.env.step(3)
        self.env.step(3)
        self.env.step(3)
        self.env.step(3)
        self.env.step(3)
        self.env.step(14)
        self.env.step(3)
        self.env.step(3)
        self.env.step(3)
        self.env.step(3)
        self.env.step(3)
        self.env.step(3)
        self.env.step(5)
        self.env.step(5)
        self.env.step(5)
        self.env.step(5)
        self.env.step(5)
        self.env.step(5)
        self.env.step(5)
        self.env.step(5)
        self.env.step(5)
        self.env.step(5)
        self.env.step(4)
        self.env.step(4)
        self.env.step(4)
        self.env.step(4)
        self.env.step(4)
        self.env.step(4)
        self.env.step(4)
        # x=105, y=148 En bas du niveau
        # self.env.step(4)
        # self.env.step(4)
        # self.env.step(4)
        # self.env.step(4)
        # self.env.step(4)
        # self.env.step(4)
        # self.env.step(4)
        # self.env.step(15)
        # self.env.step(4)
        # self.env.step(4)
        # self.env.step(4)
        # self.env.step(4)
        # self.env.step(4)
        # self.env.step(4)
        # self.env.step(4)
        # self.env.step(4)
        # self.env.step(4)
        ## (39, 148) Après la tete de mort
        # self.env.step(4)
        # self.env.step(4)
        # self.env.step(4)
        # self.env.step(4)
        # self.env.step(4)
        # self.env.step(2)
        # self.env.step(2)
        # self.env.step(2)
        # self.env.step(2)
        # self.env.step(2)
        # self.env.step(2)
        # self.env.step(2)
        # self.env.step(2)
        # self.env.step(2)
        ## (21, 192) Devant la clé en haut de l'echelle

    def setup_position_tp(self):
        self.set_agent_position(21, 192)
