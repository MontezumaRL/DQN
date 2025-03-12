# src/environment.py

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

        # Pour le suivi de la position
        self.last_position = None
        self.visited_positions = set()

        # Pour les récompenses de curiosité
        self.room_detection_grid = 10  # Grille plus fine pour détecter les salles
        self.visited_rooms = set()

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

        # Réinitialiser le suivi de position
        self.last_position = self._get_agent_position(self.frame_stack)
        self.visited_positions = {self.last_position}

        # Réinitialiser le suivi des salles
        self.visited_rooms = {self._get_room_id(self.frame_stack)}

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

        # Ajouter des récompenses de curiosité
        current_position = self._get_agent_position(self.frame_stack)
        current_room = self._get_room_id(self.frame_stack)

        # Récompense pour visiter une nouvelle position
        if current_position not in self.visited_positions:
            self.visited_positions.add(current_position)
            reward += 0.1  # Petite récompense pour l'exploration locale

        # Récompense pour découvrir une nouvelle salle
        if current_room not in self.visited_rooms:
            self.visited_rooms.add(current_room)
            reward += 1.0  # Récompense plus importante pour découvrir une nouvelle salle

        # Mettre à jour la dernière position
        self.last_position = current_position

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

    def _get_room_id(self, state):
        """
        Identifie la salle actuelle en utilisant une signature basée sur les pixels
        Cette méthode est une approximation et pourrait être améliorée
        """
        # Utiliser la dernière frame du stack
        if len(self.frame_stack) > 0:
            frame = self.frame_stack[-1]

            # Créer une signature de la salle en utilisant une grille plus grossière
            h, w = frame.shape
            grid_h, grid_w = h // self.room_detection_grid, w // self.room_detection_grid

            # Calculer l'intensité moyenne dans chaque cellule
            room_signature = []
            for i in range(grid_h):
                for j in range(grid_w):
                    cell = frame[i * self.room_detection_grid:(i + 1) * self.room_detection_grid,
                           j * self.room_detection_grid:(j + 1) * self.room_detection_grid]
                    avg_intensity = np.mean(cell)
                    # Discrétiser l'intensité pour réduire la sensibilité aux petits changements
                    room_signature.append(int(avg_intensity * 10))

            # Convertir la signature en un identifiant unique
            return tuple(room_signature)
        return (0,)  # ID par défaut

    def close(self):
        self.env.close()


class FrameSkipWrapper:
    """Wrapper pour répéter la même action plusieurs fois"""

    def __init__(self, env, skip=4):
        self.env = env
        self.skip = skip

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}

        for _ in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return state, total_reward, done, info

    def close(self):
        self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)