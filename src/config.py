class Config:
    """Configuration pour l'entraînement de l'agent DQN"""

    # Paramètres d'apprentissage
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_DECAY = 0.9995
    LEARNING_RATE = 0.0001

    # Paramètres de mémoire
    MEMORY_SIZE = 100000

    # Paramètres d'entraînement
    TARGET_UPDATE = 10
    RENDER = False  # Mettre à True pour visualiser le jeu
    NUM_EPISODES = 10000
    SAVE_INTERVAL = 100

    # Chemin de sauvegarde
    SAVE_DIR = "./output"