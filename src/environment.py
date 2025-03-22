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
        self.ale = self.env.unwrapped.ale # Pour accéder à la RAM ALE
        self.frame_stack = deque(maxlen=4)
        self.action_space = self.env.action_space
        self.n_actions = self.env.action_space.n
        self.lives = 0  # Pour suivre le nombre de vies
        self.flag_candy = True

        # Nouveaux attributs pour la pénalité de temps
        self.steps_in_episode = 0
        self.time_penalty = -0.1  # Pénalité par pas de temps
        self.max_steps = 1000     # Nombre maximum de pas par épisode

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
        self.flag_candy = True
        # Réinitialiser le compteur de pas
        self.steps_in_episode = 0

        # Initialiser le stack de frames
        for _ in range(4):
            self.frame_stack.append(state)

        # Initialiser le nombre de vies
        _, _, _, _, info = self.env.step(0)  # Action NOOP pour obtenir info
        self.lives = info.get('lives', 0)   
        #self.setup_position_tp()     

        return np.array(self.frame_stack)

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        next_state = preprocess_frame(next_state)
        self.frame_stack.append(next_state)
        # Incrémenter le compteur de pas et appliquer la pénalité de temps
        self.steps_in_episode += 1
        reward += self.time_penalty

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
        x, y = self.get_agent_position()
        
        if x <= 39 and self.flag_candy:
            reward += 50
            self.flag_candy = False
            print("candy")
        
        # Vérifier si l'épisode a atteint le nombre maximum de pas
        if self.steps_in_episode >= self.max_steps:
            reward -= 10.0  # Pénalité supplémentaire pour timeout
            done = True
        
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
        self.ale.setRAM(42,x)
        self.ale.setRAM(43,y)

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
        #self.env.step(4)
        #self.env.step(4)
        #self.env.step(4)
        #self.env.step(4)
        #self.env.step(4)
        #self.env.step(4)
        #self.env.step(4)
        #self.env.step(15)
        #self.env.step(4)
        #self.env.step(4)
        #self.env.step(4)
        #self.env.step(4)
        #self.env.step(4)
        #self.env.step(4)
        #self.env.step(4)
        #self.env.step(4)
        #self.env.step(4)
        ## (39, 148) Après la tete de mort
        #self.env.step(4)
        #self.env.step(4)
        #self.env.step(4)
        #self.env.step(4)
        #self.env.step(4)
        #self.env.step(2)
        #self.env.step(2)
        #self.env.step(2)
        #self.env.step(2)
        #self.env.step(2)
        #self.env.step(2)
        #self.env.step(2)
        #self.env.step(2)
        #self.env.step(2)
        ## (21, 192) Devant la clé en haut de l'echelle
    
    def setup_position_tp(self):
        self.set_agent_position(21,192)
