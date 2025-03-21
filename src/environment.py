# fichier dqn/src/environment.py
import gymnasium
from collections import deque
import numpy as np
import torch
from .utils import preprocess_frame
import matplotlib.pyplot as plt
import ale_py


class MontezumaEnvironment:
    def __init__(self, render_mode):
        self.env = gymnasium.make("ALE/MontezumaRevenge-v5", render_mode=render_mode)
        self.frame_stack = deque(maxlen=4)
        self.action_space = self.env.action_space
        self.n_actions = self.env.action_space.n
        self.lives = 0  # Pour suivre le nombre de vies
        self.grid_size = 8

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

        # Initialiser le stack de frames
        for _ in range(4):
            self.frame_stack.append(state)

        # Initialiser le nombre de vies
        _, _, _, _, info = self.env.step(0)  # Action NOOP pour obtenir info
        self.lives = info.get('lives', 0)

        return np.array(self.frame_stack)

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        next_state = preprocess_frame(next_state)
        self.frame_stack.append(next_state)

        done = terminated or truncated

        # Vérifier si une vie a été perdue
        current_lives = info.get('lives', 0)
        life_lost = current_lives < self.lives
        self.lives = current_lives

        if life_lost:
            reward -= 10.0
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

    def _get_agent_position(self, state):
        """
        Extrait une position approximative de l'agent à partir de l'état
        Cette implémentation est simplifiée et utilise une grille discrète
        """
        # Utiliser la dernière frame du stack
        if len(self.frame_stack) > 0:
            frame = self.frame_stack[-1]

            # Discrétiser l'espace en une grille
            h, w = frame.shape
            grid_h, grid_w = h // self.grid_size, w // self.grid_size

            # Calculer l'intensité moyenne dans chaque cellule
            intensity_grid = np.zeros((grid_h, grid_w))
            for i in range(grid_h):
                for j in range(grid_w):
                    cell = frame[i * self.grid_size:(i + 1) * self.grid_size,
                           j * self.grid_size:(j + 1) * self.grid_size]
                    intensity_grid[i, j] = np.mean(cell)

            # Trouver la cellule avec la plus grande différence d'intensité
            # (approximation grossière de la position de l'agent)
            flat_idx = np.argmax(intensity_grid)
            pos_y, pos_x = np.unravel_index(flat_idx, intensity_grid.shape)

            return (pos_x, pos_y)
        return (0, 0)  # Position par défaut

    def close(self):
        self.env.close()
