# fichier: dqn/src/environment.py
import gymnasium as gym # MODIFICATION: Importer gym
from gymnasium import spaces # MODIFICATION: Importer spaces
from collections import deque
import numpy as np
import torch
from .utils import preprocess_frame # Import depuis utils
import matplotlib.pyplot as plt
import ale_py
import time
import random

class MontezumaEnvironment(gym.Env):
    """
    Environnement personnalisé pour Montezuma's Revenge.
    - Applique le prétraitement des frames (gris, 42x42, normalisation).
    - Gère un stack de 4 frames comme état.
    - Gère la perte de vie comme une terminaison d'épisode.
    - Permet l'accès à la RAM ALE.
    - Hérite de gym.Env pour être compatible avec les wrappers Gymnasium.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None, seed=None, max_episode_steps=1000, step_limit_penalty=-1.0 , curriculum_config=None):
        """
        Initialise l'environnement.

        Args:
            render_mode (str, optional): Mode de rendu ('human', 'rgb_array', None).
            seed (int, optional): Graine pour la reproductibilité.
        """
        super().__init__()

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4, 42, 42), # Format (C, H, W)
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.internal_env = gym.make("ALE/MontezumaRevenge-v5", render_mode=self.render_mode)

        self.action_space = self.internal_env.action_space
        self.n_actions = self.action_space.n 

        try:
            self.ale = self.internal_env.unwrapped.ale
        except AttributeError:
             print("Warning: Could not get ALE interface via self.internal_env.unwrapped.ale. RAM access might fail.")
             self.ale = None

        self.frame_stack = deque(maxlen=4) # Stack de frames
        self.lives = 0                     # Vies restantes
        self._initial_seed = seed          # Stocker la seed initiale fournie

        self.max_steps = max_episode_steps
        self.step_limit_penalty = step_limit_penalty
        self.episode_step_count = 0
        self.step_penalty_value = -0.001 # Pénalité par pas pour encourager l'agent à avancer
        
        self.curriculum_config = curriculum_config if curriculum_config else {}
        self.teleport_locations = self.curriculum_config.get("locations", {})


    def reset(self, seed=None, options=None):
        """
        Réinitialise l'environnement pour un nouvel épisode.

        Args:
            seed (int, optional): Graine spécifique pour cet épisode. Si None, utilise une seed dérivée de la seed initiale.
            options (dict, optional): Options pour l'environnement interne.

        Returns:
            tuple: (np.ndarray: état initial stacké, dict: infos initiales)
        """
        super().reset(seed=seed)

        if seed is None:
            seed = self._initial_seed

        state, info = self.internal_env.reset(seed=seed, options=options)

        # Prétraiter la première frame et remplir le stack
        processed_state = preprocess_frame(state)
        for _ in range(4):
            self.frame_stack.append(processed_state)

        self.lives = info.get('lives', 0)
        self.episode_step_count = 0
        
        # --- Logique du Curriculum ---
        apply_teleport = False
        location_key = "normal" # Défaut si pas de téléportation

        if self.curriculum_config.get("enabled", False) and self.teleport_locations:
            prob_teleport = self.curriculum_config.get("teleport_prob", 0.0)
            if random.random() < prob_teleport:
                apply_teleport = True
                location_key = random.choice(list(self.teleport_locations.keys()))
                teleport_pos = self.teleport_locations[location_key]

                # 1. Téléporter
                self.set_agent_position(teleport_pos[0], teleport_pos[1])

                # 2. Stabiliser l'état et remplir le frame stack avec des NOOPs
                num_noop_steps = 5 # Nombre de pas à vide, peut nécessiter ajustement
                self.frame_stack.clear()
                for i in range(self.frame_stack.maxlen + num_noop_steps -1):
                    next_state, _, terminated, truncated, step_info = self.internal_env.step(0)
                    processed_state = preprocess_frame(next_state)
                    self.frame_stack.append(processed_state)
                    self.lives = step_info.get('lives', self.lives)

                    if terminated or truncated:
                         print(f"Warning: Episode ended during NOOP stabilization after teleport to {location_key}. Resetting again.")
                         return self.reset(seed=seed, options=options) # Appel récursif simple

        if not apply_teleport:
            processed_state = preprocess_frame(state)
            if len(self.frame_stack) < self.frame_stack.maxlen:
                self.frame_stack.clear()
                for _ in range(self.frame_stack.maxlen):
                    self.frame_stack.append(processed_state)

        initial_obs = np.array(self.frame_stack, dtype=np.float32)

        expected_shape = self.observation_space.shape
        if initial_obs.shape != expected_shape:
            print(f"Error: Observation shape mismatch after reset! Expected {expected_shape}, got {initial_obs.shape}. Resetting standard.")
            self.frame_stack.clear()
            state, info = self.internal_env.reset(seed=seed, options=options)
            self.lives = info.get('lives', 0); self.episode_step_count = 0
            processed_state = preprocess_frame(state)
            for _ in range(self.frame_stack.maxlen): self.frame_stack.append(processed_state)
            initial_obs = np.array(self.frame_stack, dtype=np.float32)
            location_key = "error_fallback"
            apply_teleport = False


        assert self.observation_space.contains(initial_obs), f"Initial observation does not match observation space. Shape: {initial_obs.shape}"

        if info is None: info = {}
        info['teleported_start'] = apply_teleport
        info['start_location'] = location_key

        return initial_obs, info

    def step(self, action):
        """
        Exécute une action dans l'environnement.

        Args:
            action (int): L'action à exécuter.

        Returns:
            tuple: (np.ndarray: état suivant stacké, float: récompense extrinsèque,
                    bool: terminated, bool: truncated, dict: infos)
                   Conforme à l'API gym.Env.
        """
        if self.episode_step_count >= self.max_steps:
             print(f"Warning: step() called after {self.max_steps} steps. Returning terminated state.")
             current_stack_obs = np.array(self.frame_stack, dtype=np.float32) # Etat actuel
             return current_stack_obs, 0.0, True, False, {"error": "Step called after limit reached"}

        next_state, reward, terminated, truncated, info = self.internal_env.step(action)

        # Prétraiter la nouvelle frame
        processed_next_state = preprocess_frame(next_state)
        self.frame_stack.append(processed_next_state)

        # Récupérer l'état actuel du stack
        current_stack_obs = np.array(self.frame_stack, dtype=np.float32)
        assert self.observation_space.contains(current_stack_obs), "Step observation does not match observation space"

        self.episode_step_count += 1

        # Récompense extrinsèque
        extrinsic_reward = float(reward)

        extrinsic_reward += self.step_penalty_value

        # Logique de perte de vie : considérer comme 'terminated'
        current_lives = info.get('lives', 0)
        life_lost = current_lives < self.lives
        if life_lost and self.lives > 0:
            extrinsic_reward -= 1.0 # Appliquer pénalité
            terminated = True 

        # Mettre à jour le nombre de vies pour le prochain pas
        self.lives = current_lives

        if self.episode_step_count >= self.max_steps and not terminated:
            extrinsic_reward += self.step_limit_penalty # Ajouter la pénalité négative
            terminated = True # Forcer la terminaison car la limite de pas est atteinte
        
        if extrinsic_reward > 0:
            terminated = True

        return current_stack_obs, extrinsic_reward, terminated, truncated, info

    def close(self):
        """Ferme l'environnement interne."""
        print("Closing internal Montezuma environment.")
        self.internal_env.close()

    def get_state_tensor(self, device):
        """Convertit le stack de frames actuel en tenseur PyTorch."""
        state_np = np.array(self.frame_stack, dtype=np.float32)
        return torch.from_numpy(state_np).unsqueeze(0).to(device)

    def display_frame_stack(self, save_path=None, show=True, delay=0.02):
        """Affiche ou sauvegarde les 4 frames de la stack actuelle."""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for i, frame in enumerate(self.frame_stack):
            axes[i].imshow(frame, cmap='gray')
            axes[i].set_title(f'Frame {i + 1}')
            axes[i].axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show(block=False)
            plt.pause(delay)
        plt.close(fig)

    def get_agent_position(self):
        """Récupère la position (x, y) de l'agent depuis la RAM ALE."""
        if self.ale is None:
            print("Warning: ALE interface not available for get_agent_position.")
            return -1, -1
        try:
            ram = self.ale.getRAM()
            x_coord = int(ram[42])
            y_coord = int(ram[43])
            return x_coord, y_coord
        except Exception as e:
            print(f"Warning: Error accessing RAM for position: {e}")
            return -1, -1

    def set_agent_position(self, x, y):
        """Définit la position (x, y) de l'agent dans la RAM ALE."""
        if self.ale is None:
            print("Warning: ALE interface not available for set_agent_position.")
            return
        try:
            x_safe = max(0, min(255, int(x)))
            y_safe = max(0, min(255, int(y)))
            self.ale.setRAM(42, x_safe)
            self.ale.setRAM(43, y_safe)
        except Exception as e:
            print(f"Warning: Error writing RAM for position: {e}")

    def setup_position_tp(self):
        print("Attempting to teleport agent via RAM manipulation...")
        target_x, target_y = 21, 192
        self.set_agent_position(target_x, target_y)
        # Faire quelques pas NOOP pour que l'état se stabilise / soit observé
        for _ in range(4):
            self.internal_env.step(0) # Action NOOP# --- Curriculum Learning Config ---
            USE_CURRICULUM = True # Activer/Désactiver globalement
            