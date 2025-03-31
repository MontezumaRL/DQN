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

# MODIFICATION: Faire hériter la classe de gym.Env
class MontezumaEnvironment(gym.Env):
    """
    Environnement personnalisé pour Montezuma's Revenge.
    - Applique le prétraitement des frames (gris, 42x42, normalisation).
    - Gère un stack de 4 frames comme état.
    - Gère la perte de vie comme une terminaison d'épisode.
    - Permet l'accès à la RAM ALE.
    - Hérite de gym.Env pour être compatible avec les wrappers Gymnasium.
    """
    # Optionnel: Métadonnées pour Gymnasium
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None, seed=None, max_episode_steps=2000, step_limit_penalty=-1.0 , curriculum_config=None):
        """
        Initialise l'environnement.

        Args:
            render_mode (str, optional): Mode de rendu ('human', 'rgb_array', None).
            seed (int, optional): Graine pour la reproductibilité.
        """
        super().__init__() # MODIFICATION: Appel constructeur parent gym.Env

        # --- Définition des espaces (Requis par gym.Env) ---
        # Espace d'observation: 4 frames 42x42 en float32 normalisé [0,1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4, 42, 42), # Format (C, H, W)
            dtype=np.float32
        )

        # L'espace d'action sera défini après la création de l'env interne
        # self.action_space = spaces.Discrete(...) # Sera défini ci-dessous

        # --- Création de l'environnement Gym interne ---
        self.render_mode = render_mode
        # MODIFICATION: Renommer self.env en self.internal_env
        self.internal_env = gym.make("ALE/MontezumaRevenge-v5", render_mode=self.render_mode)

        # --- Définir l'action_space basé sur l'env interne (Requis par gym.Env) ---
        self.action_space = self.internal_env.action_space
        self.n_actions = self.action_space.n # Garder pour l'agent

        # Accès à l'interface ALE
        try:
            # MODIFICATION: Utiliser self.internal_env
            self.ale = self.internal_env.unwrapped.ale
        except AttributeError:
             print("Warning: Could not get ALE interface via self.internal_env.unwrapped.ale. RAM access might fail.")
             self.ale = None

        # --- Variables d'état internes ---
        self.frame_stack = deque(maxlen=4) # Stack de frames
        self.lives = 0                     # Vies restantes
        self._initial_seed = seed          # Stocker la seed initiale fournie

        # --- AJOUT: Variables pour la limite de pas ---
        self.max_steps = max_episode_steps
        self.step_limit_penalty = step_limit_penalty
        self.episode_step_count = 0
        
        # --- AJOUT: Initialisation Curriculum ---
        self.curriculum_config = curriculum_config if curriculum_config else {}
        self.teleport_locations = self.curriculum_config.get("locations", {})
        # Exemple de structure pour self.teleport_locations (à remplir par vous):
        # self.teleport_locations = {
        #     "near_key": (78, 233),   # Coordonnées à ajuster !
        #     "bottom_ladder1": (21, 192), # Coordonnées à ajuster !
        #     "after_skull_jump": (..., ...), # Coordonnées à ajuster !
        #     # Ajoutez autant de points que nécessaire
        # }


    def reset(self, seed=None, options=None):
        """
        Réinitialise l'environnement pour un nouvel épisode.

        Args:
            seed (int, optional): Graine spécifique pour cet épisode. Si None, utilise une seed dérivée de la seed initiale.
            options (dict, optional): Options pour l'environnement interne.

        Returns:
            tuple: (np.ndarray: état initial stacké, dict: infos initiales)
        """
        # MODIFICATION: Appel à super().reset() pour gérer la seed interne de gym.Env
        super().reset(seed=seed)

        # Logique de gestion de seed améliorée: utilise la seed passée, ou la seed initiale
        if seed is None:
            seed = self._initial_seed
        # Si une seed est fournie, elle prime pour ce reset spécifique
        # Note: gym.Env gère aussi une RNG interne, super().reset(seed=seed) s'en occupe.

        # Réinitialiser l'environnement interne avec la seed
        # MODIFICATION: Utiliser self.internal_env
        state, info = self.internal_env.reset(seed=seed, options=options)

        # Prétraiter la première frame et remplir le stack
        processed_state = preprocess_frame(state)
        for _ in range(4):
            self.frame_stack.append(processed_state)

        # Initialiser le nombre de vies
        self.lives = info.get('lives', 0)

        self.episode_step_count = 0
        
        # --- Logique du Curriculum ---
        apply_teleport = False
        location_key = "normal" # Défaut si pas de téléportation

        if self.curriculum_config.get("enabled", False) and self.teleport_locations:
            prob_teleport = self.curriculum_config.get("teleport_prob", 0.0)
            if random.random() < prob_teleport:
                apply_teleport = True
                # Choisir une location (ici, aléatoire parmi celles définies)
                location_key = random.choice(list(self.teleport_locations.keys()))
                teleport_pos = self.teleport_locations[location_key]
                # print(f"Curriculum: Teleporting to {location_key} at {teleport_pos}") # Décommenter pour debug

                # 1. Téléporter
                self.set_agent_position(teleport_pos[0], teleport_pos[1])

                # 2. Stabiliser l'état et remplir le frame stack avec des NOOPs
                num_noop_steps = 5 # Nombre de pas à vide, peut nécessiter ajustement
                # Vider le stack actuel qui contient l'état du reset initial
                self.frame_stack.clear()
                temp_state = state # Garder une référence temporaire si besoin
                for i in range(self.frame_stack.maxlen + num_noop_steps -1): # Assurer assez d'étapes pour remplir le stack
                    # Utiliser action 0 (NOOP) sur l'environnement interne
                    next_state, _, terminated, truncated, step_info = self.internal_env.step(0)
                    processed_state = preprocess_frame(next_state)
                    self.frame_stack.append(processed_state) # Remplir le stack avec les frames post-téléportation
                    self.lives = step_info.get('lives', self.lives) # Mettre à jour les vies

                    # Gérer une mort/fin d'épisode pendant les NOOPs (très peu probable mais sait-on jamais)
                    if terminated or truncated:
                         print(f"Warning: Episode ended during NOOP stabilization after teleport to {location_key}. Resetting again.")
                         # Si l'épisode se termine pendant la stabilisation, il faut recommencer le reset.
                         # C'est un cas rare mais peut arriver si la position de téléport est fatale.
                         # On pourrait choisir de ne pas téléporter dans ce cas pour éviter une boucle.
                         # Pour simplifier ici, on retourne le résultat du reset standard.
                         # Note: une meilleure gestion pourrait être nécessaire si cela arrive souvent.
                         return self.reset(seed=seed, options=options) # Appel récursif simple

                # print(f"Curriculum: NOOP steps completed after teleport to {location_key}.") # Décommenter pour debug


        if not apply_teleport:
            # Pas de téléportation OU échec de la stabilisation : remplir le stack normalement
            processed_state = preprocess_frame(state)
            # Assurer que le stack est plein même si apply_teleport était True mais a échoué (cas edge)
            if len(self.frame_stack) < self.frame_stack.maxlen:
                self.frame_stack.clear()
                for _ in range(self.frame_stack.maxlen):
                    self.frame_stack.append(processed_state)

        # Construire l'observation initiale finale
        initial_obs = np.array(self.frame_stack, dtype=np.float32)

        # Vérification finale de la forme (importante après la logique complexe)
        expected_shape = self.observation_space.shape
        if initial_obs.shape != expected_shape:
            print(f"Error: Observation shape mismatch after reset! Expected {expected_shape}, got {initial_obs.shape}. Resetting standard.")
             # Fallback vers un reset standard sans curriculum en cas d'erreur de forme
            self.frame_stack.clear()
            state, info = self.internal_env.reset(seed=seed, options=options)
            self.lives = info.get('lives', 0); self.episode_step_count = 0
            processed_state = preprocess_frame(state)
            for _ in range(self.frame_stack.maxlen): self.frame_stack.append(processed_state)
            initial_obs = np.array(self.frame_stack, dtype=np.float32)
            location_key = "error_fallback"
            apply_teleport = False


        assert self.observation_space.contains(initial_obs), f"Initial observation does not match observation space. Shape: {initial_obs.shape}"

        # --- AJOUT: Informations pour le logging ---
        # S'assurer que 'info' est un dictionnaire
        if info is None: info = {}
        info['teleported_start'] = apply_teleport
        info['start_location'] = location_key # Contient "normal" ou la clé de la location
        # ------------------------------------------

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
        # Exécuter l'action dans l'environnement Gym interne
        # MODIFICATION: Utiliser self.internal_env
        # Vérifier si l'épisode ne devrait pas déjà être terminé (sécurité)
        if self.episode_step_count >= self.max_steps:
             # Normalement, cela ne devrait pas arriver si l'appelant respecte terminated/truncated,
             # mais c'est une sécurité. Retourne un état valide mais marque comme terminé.
             print(f"Warning: step() called after {self.max_steps} steps. Returning terminated state.")
             current_stack_obs = np.array(self.frame_stack, dtype=np.float32) # Etat actuel
             # Retourne 0 reward, terminated=True, truncated=False (car on gère comme termination)
             # et info vide ou la dernière info connue.
             return current_stack_obs, 0.0, True, False, {"error": "Step called after limit reached"}

        next_state, reward, terminated, truncated, info = self.internal_env.step(action)

        # Prétraiter la nouvelle frame
        processed_next_state = preprocess_frame(next_state)
        self.frame_stack.append(processed_next_state)

        # Récupérer l'état actuel du stack
        current_stack_obs = np.array(self.frame_stack, dtype=np.float32)
        assert self.observation_space.contains(current_stack_obs), "Step observation does not match observation space"

# --- AJOUT: Incrémenter le compteur de pas ---
        self.episode_step_count += 1
        # --------------------------------------------

        # Récompense extrinsèque
        extrinsic_reward = float(reward)

        # Logique de perte de vie : considérer comme 'terminated'
        current_lives = info.get('lives', 0)
        life_lost = current_lives < self.lives
        if life_lost and self.lives > 0:
            extrinsic_reward -= 1.0 # Appliquer pénalité
            terminated = True     # MODIFICATION: Traiter la perte de vie comme terminaison

        # Mettre à jour le nombre de vies pour le prochain pas
        self.lives = current_lives

        if self.episode_step_count >= self.max_steps and not terminated:
            extrinsic_reward += self.step_limit_penalty # Ajouter la pénalité négative
            terminated = True # Forcer la terminaison car la limite de pas est atteinte

        # La troncature (ex: par limite de temps) est gérée soit par l'env interne,
        # soit par le wrapper TimeLimit ajouté dans train.py

        # Retourner les 5 éléments requis par l'API gym.Env step
        return current_stack_obs, extrinsic_reward, terminated, truncated, info

    def close(self):
        """Ferme l'environnement interne."""
        print("Closing internal Montezuma environment.")
        # MODIFICATION: Utiliser self.internal_env
        self.internal_env.close()

    # --- Méthodes spécifiques à cet environnement ---
    # Ces méthodes sont conservées mais ne font pas partie de l'API gym.Env standard

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
            # print(f"Frame stack saved to {save_path}") # Optionnel: réduire verbosité
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

    # --- Méthodes de positionnement spécifiques (inchangées) ---
    def setup_position_human(self):
        print("Attempting to reach specific position using pre-defined actions...")
        # ... (ta séquence d'actions .step() sur self.internal_env) ...
        pass

    def setup_position_tp(self):
        print("Attempting to teleport agent via RAM manipulation...")
        target_x, target_y = 21, 192
        self.set_agent_position(target_x, target_y)
        # Faire quelques pas NOOP pour que l'état se stabilise / soit observé
        for _ in range(4):
             # MODIFICATION: Utiliser self.internal_env
            self.internal_env.step(0) # Action NOOP# --- Curriculum Learning Config ---
            USE_CURRICULUM = True # Activer/Désactiver globalement
            