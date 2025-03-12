class Config:
    """Configuration pour l'entraînement de l'agent DQN"""

    # Paramètres d'apprentissage
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_DECAY = 0.995
    LEARNING_RATE = 0.0001

    # Paramètres de mémoire
    MEMORY_SIZE = 100000

    # Paramètres d'entraînement
    TARGET_UPDATE = 10
    RENDER = True  # Mettre à True pour visualiser le jeu
    NUM_EPISODES = 10000
    SAVE_INTERVAL = 100

    # Chemin de sauvegarde
    SAVE_DIR = "output"

    # Paramètres pour l'exploration intrinsèque
    INTRINSIC_REWARD_SCALE = 0.01
    EMBEDDING_BUFFER_SIZE = 10000
    NOVELTY_LR = 0.0001

    # Paramètres pour Prioritized Experience Replay
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    PER_BETA_FRAMES = 100000

    # Paramètres pour l'entraînement
    N_STEP_RETURNS = 3  # Pour les retours à n étapes
    FRAME_SKIP = 4  # Répéter la même action plusieurs fois

    # Paramètres pour l'exploration
    EXPLORATION_FRACTION = 0.1  # Fraction de l'entraînement pour l'exploration
    FINAL_EXPLORATION = 0.01  # Valeur finale d'epsilon