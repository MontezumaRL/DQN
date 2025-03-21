import os
import time

import torch
from ..agents.dqn import DQNAgent
from ..config import Config
from ..environment import MontezumaEnvironment


def train_montezuma():
    # Création de l'environnement
    env = MontezumaEnvironment(render_mode="human" if Config.RENDER else None)

    # Détection du device (CPU ou GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialisation de l'agent
    agent = DQNAgent((4, 84, 84), env.n_actions, device, Config)

    # Initialisation des variables de suivi
    episode_rewards = []

    # Boucle d'entraînement
    for episode in range(Config.NUM_EPISODES):
        # Réinitialisation de l'environnement
        state = env.reset()
        episode_reward = 0
        episode_start_time = time.time()
        nb_steps = 0

        # Boucle d'un épisode
        while True:
            # Sélection de l'action (epsilon-greedy)
            action = agent.select_action(state)

            # Exécution de l'action
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            nb_steps += 1

            # Stockage de l'expérience dans le buffer
            agent.store_transition(state, action, reward, next_state, done)

            # Passage à l'état suivant
            state = next_state

            # Apprentissage
            agent.learn()

            # Fin de l'épisode
            if done:
                break

        # Mise à jour du réseau cible
        if episode % Config.TARGET_UPDATE == 0:
            agent.update_target_network()

        # Mise à jour d'epsilon
        agent.update_epsilon()

        # Calcul du temps d'épisode
        episode_duration = time.time() - episode_start_time

        # Enregistrement des récompenses
        episode_rewards.append(episode_reward)

        # Affichage des statistiques
        print(f"Episode {episode}, Reward: {episode_reward}, "
              f"Epsilon: {agent.epsilon:.2f}, Duration: {episode_duration:.2f}s, "
              f"Steps: {nb_steps}")

        # Sauvegarde périodique du modèle
        if episode % Config.SAVE_INTERVAL == 0:
            os.makedirs(Config.SAVE_DIR, exist_ok=True)
            agent.save(f"{Config.SAVE_DIR}/montezuma_dqn_ep{episode}.pth", episode, episode_rewards)
            print(f"Model saved at episode {episode}")

    # Fermeture de l'environnement
    env.close()

    return agent.policy_net