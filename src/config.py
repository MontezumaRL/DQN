# fichier: dqn/src/config.py
import torch

# Configuration de l'environnement
ENV_NAME = "ALE/MontezumaRevenge-v5"
FRAME_STACK = 4
INPUT_SHAPE = (FRAME_STACK, 42, 42) # (C, H, W)

# Hyperparamètres DDQN
GAMMA = 0.99                # Facteur de discount
BATCH_SIZE = 32             # Taille du batch pour l'entraînement
BUFFER_SIZE = 100000        # Capacité max du replay buffer (100k pour commencer, viser 1M+)
MIN_BUFFER_SIZE = 10000     # Taille min avant de commencer l'entraînement (10k-50k)
TARGET_UPDATE_FREQ = 8000   # Fréquence de mise à jour du target network (en nombre de pas)
LEARNING_RATE_DQN = 1e-5    # Taux d'apprentissage pour DQN (Adam optimizer) avant changement de spawn 1e-4

# Exploration (Epsilon-greedy)
EPSILON_START = 0.3         # Valeur initiale d'epsilon
EPSILON_FINAL = 0.01        # Valeur finale d'epsilon (pas 0 pour garder un peu d'explo)
EPSILON_DECAY_FRAMES = 1000000 # Nombre de frames sur lesquelles epsilon décroît linéairement

# Hyperparamètres RND
USE_RND = False              # Activer ou non RND
RND_OUTPUT_DIM = 512        # Dimension de l'embedding RND
RND_LR = 5e-5             # Taux d'apprentissage pour le prédicteur RND
INTRINSIC_REWARD_SCALE = 0.5 # Facteur beta pour pondérer la récompense intrinsèque (à tuner!)
RND_OBS_CLIP = 5.0          # Clipping pour les observations normalisées RND
RND_REWARD_CLIP = 1.0       # Clipping pour les récompenses intrinsèques normalisées

# Configuration entraînement
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_FRAMES = 10000000       # Nombre total de frames pour l'entraînement (10M pour commencer)
LOG_FREQ = 1000             # Fréquence de log (en nombre de pas)
SAVE_FREQ = 10000          # Fréquence de sauvegarde du modèle (en nombre de pas)
MODEL_SAVE_PATH = "dqn_montezuma_model.pth"
RND_SAVE_PATH = "rnd_montezuma_model.pth"
OPTIM_SAVE_PATH = "optimizers_montezuma.pth"
NORM_SAVE_PATH = "normalizers_montezuma.pkl"

SEED = 57                   # Graine pour la reproductibilité

# Normalisation des récompenses RND (coefficient gamma_r pour le RMS de la récompense)
# Normalise r_i = (r_i_raw / (std_r_i + eps))
INTRINSIC_REWARD_NORM_GAMMA = GAMMA # Utilise le même gamma que DQN par défaut

# --- Curriculum Learning Config ---
USE_CURRICULUM = True # Activer/Désactiver globalement
CURRICULUM_CONFIG = {
    "enabled": USE_CURRICULUM,
    "teleport_prob": 1,  # Probabilité de téléporter au début d'un épisode
    "locations": {
        # --- REMPLISSEZ AVEC VOS COORDONNÉES PRÉCISES ---
        #"bottom_ladder_middle": (77, 192),   # Exemple: en bas de l'échelle du milieu
        "top_ladder_right": (133, 192),    # Exemple: en haut de l'échelle à droite
        "bottom_ladder_right": (105, 148),   # Exemple: au pied de la première échelle
        #"after_skull_jump": (39, 148),# Exemple: après le saut du crâne
        #"near_key": (21, 192),      # Exemple: juste en dessous de la clé
    }
}