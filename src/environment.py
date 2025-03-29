# fichier: dqn/src/environment.py
import gymnasium
from collections import deque
import numpy as np
import torch
# Modifie l'import:
from .utils import preprocess_frame # <--- CHANGEMENT ICI
import matplotlib.pyplot as plt
import ale_py

class MontezumaEnvironment:
    def __init__(self, render_mode=None, seed=None): # Ajout de seed ici
        # Ajout gestion du seed pour la reproductibilité
        self.env = gymnasium.make("ALE/MontezumaRevenge-v5", render_mode=render_mode)
        # Passer le seed à l'environnement si fourni
        # Note: reset() est le meilleur endroit pour le seed, mais on le met ici aussi
        if seed is not None:
             self.env.reset(seed=seed)

        self.ale = self.env.unwrapped.ale # Pour accéder à la RAM ALE
        self.frame_stack = deque(maxlen=4)
        self.action_space = self.env.action_space
        self.n_actions = self.env.action_space.n
        self.lives = 0
        self.flag_candy = True # Tu pourrais vouloir enlever ça si RND suffit

        # Pénalité de temps (peut être contre-productif avec RND, à tester)
        # self.steps_in_episode = 0
        # self.time_penalty = -0.01 # Réduit la pénalité
        self.max_steps = 18000 # ~5 minutes de jeu max par épisode (Atari tourne à 60fps, step saute 4 frames)

    def reset(self, seed=None, options=None):
        # Il est préférable de gérer le seed ici
        state_info = self.env.reset(seed=seed, options=options)
        state = state_info[0]
        info = state_info[1] # Récupérer les infos initiales

        processed_state = preprocess_frame(state)
        self.flag_candy = True
        # self.steps_in_episode = 0 # Réinitialise le compteur de pas

        for _ in range(4):
            self.frame_stack.append(processed_state)

        # Initialiser le nombre de vies à partir des infos de reset
        self.lives = info.get('lives', 0)
        # print(f"Reset complete. Initial lives: {self.lives}")

        # self.setup_position_tp() # Décommente si tu veux tester depuis une position spécifique

        return np.array(self.frame_stack), info # Retourne aussi les infos

    def step(self, action):
        # On prend l'action, gymnasium gère le frame skip par défaut
        next_state, reward, terminated, truncated, info = self.env.step(action)

        # self.steps_in_episode += 1

        processed_next_state = preprocess_frame(next_state)
        self.frame_stack.append(processed_next_state)

        # Gérer la fin d'épisode (terminated ou truncated)
        done = terminated or truncated

        # Récompense extrinsèque (du jeu)
        extrinsic_reward = float(reward)

        # Pénalité pour perte de vie (important pour Montezuma)
        current_lives = info.get('lives', 0)
        life_lost = current_lives < self.lives
        if life_lost and self.lives > 0: # Évite la pénalité au tout début si on commence avec 0 vie
            #print(f"Life lost! Current lives: {current_lives}, Previous lives: {self.lives}")
            extrinsic_reward -= 1.0 # Pénalité forte pour la perte de vie (normalisée)
            # On considère souvent une perte de vie comme la fin d'un "mini-épisode" pour l'update du Q-learning
            # done = True # Commente ça si tu veux que l'agent continue après une perte de vie

        self.lives = current_lives

        # Limite de pas (gérée par truncated maintenant avec TimeLimit wrapper de gym)
        # if self.steps_in_episode >= self.max_steps:
        #     # extrinsic_reward -= 1.0 # Pénalité time out (optionnel)
        #     # done = True # truncated s'en occupe
        #     pass

        # Pénalité de temps (Optionnel, peut interférer avec RND)
        # extrinsic_reward += self.time_penalty

        # Bonus custom (Optionnel, RND devrait gérer ça)
        # x, y = self.get_agent_position()
        # if x <= 39 and self.flag_candy:
        #    extrinsic_reward += 1.0 # Normalise le bonus aussi
        #    self.flag_candy = False
        #    print("candy collected (custom bonus)")

        # Si un gros score est atteint (clé?), on peut arrêter l'épisode (optionnel)
        # if reward > 100: # ex: 100 pour la clé
        #    print("Key collected maybe?")
        #    # done = True # Optionnel

        # Retourne l'état stacké, la récompense *extrinsèque*, done, et info
        return np.array(self.frame_stack), extrinsic_reward, done, info

    # Reste de tes méthodes (get_state_tensor, display_frame_stack, etc.)
    # ... (elles semblent correctes, garde-les)
    def get_state_tensor(self, device):
        # Assure-toi que le tensor est C,H,W et de type float
        state_np = np.array(self.frame_stack, dtype=np.float32)
        return torch.from_numpy(state_np).unsqueeze(0).to(device)

    def display_frame_stack(self, save_path="frame_stack.png"):
        """Affiche les 4 frames de la stack actuelle ou les sauvegarde dans un fichier"""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for i, frame in enumerate(self.frame_stack):
            axes[i].imshow(frame, cmap='gray')
            axes[i].set_title(f'Frame {i + 1}')
            axes[i].axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Frame stack sauvegardée dans {save_path}")
        else:
            plt.show()
        plt.close(fig) # Ferme la figure

    def get_agent_position(self):
        """Récupère la position de l'agent dans la RAM ALE"""
        try:
            ram = self.ale.getRAM()
            # Ces adresses RAM sont spécifiques à Montezuma's Revenge
            # Vérifie si elles sont correctes pour ta version
            return int(ram[42]), int(ram[43])
        except Exception as e:
            print(f"Erreur accès RAM: {e}")
            return -1, -1 # Valeur par défaut en cas d'erreur

    def set_agent_position(self, x, y):
        """Déplace l'agent à la position (x, y) dans la RAM ALE"""
        try:
            self.ale.setRAM(42, x)
            self.ale.setRAM(43, y)
        except Exception as e:
            print(f"Erreur écriture RAM: {e}")

    def close(self):
        self.env.close()

    # Tes fonctions setup_position_* restent inchangées
    def setup_position_human(self):
        # ... (ton code)
        pass

    def setup_position_tp(self):
        # ... (ton code)
        pass