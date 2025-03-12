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

    def reset(self):
        state = self.env.reset()[0]
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

        # Exemple de log info:
        #{'lives': 6, 'episode_frame_number': 8, 'frame_number': 1040}
        #{'lives': 6, 'episode_frame_number': 12, 'frame_number': 1044}
        #{'lives': 6, 'episode_frame_number': 16, 'frame_number': 1048}
        #{'lives': 6, 'episode_frame_number': 20, 'frame_number': 1052}
        #{'lives': 6, 'episode_frame_number': 24, 'frame_number': 1056}

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

    def close(self):
        self.env.close()
