class DuelingDQNConfig:
    # Paramètres d'apprentissage
    RENDER = True
    BATCH_SIZE = 32
    LEARNING_RATE = 0.00025
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.9995

    # Paramètres du buffer
    REPLAY_SIZE = 100000
    REPLAY_INITIAL = 10000

    # Paramètres de l'agent
    N_STEPS = 3
    TAU = 0.005  # Pour la mise à jour douce
    GRAD_CLIP = 10.0
    WEIGHT_DECAY = 1e-5
    TARGET_UPDATE = 1000

    # Paramètres d'entraînement
    NUM_EPISODES = 10000
    MAX_STEPS = 1000
    LEARN_EVERY = 4
    UPDATE_TARGET_EVERY = 1000

    # Paramètres de sauvegarde
    SAVE_INTERVAL = 100
    SAVE_DIR = "output/ddqn"
    MAX_EPISODE_DURATION = 10  # Durée maximale d'un épisode en secondes