# src/training/train.py

import os
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

import torch
from ..agents.dqn import DQNAgent
from ..config import Config
from ..environment import MontezumaEnvironment, FrameSkipWrapper


def train_montezuma():
    # Création de l'environnement
    env = FrameSkipWrapper(MontezumaEnvironment(render_mode="human" if Config.RENDER else None),
                           skip=Config.FRAME_SKIP)

    # Détection du device (CPU ou GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialisation de l'agent
    agent = DQNAgent((4, 84, 84), env.n_actions, device, Config)

    # Initialisation des variables de suivi
    episode_rewards = []
    recent_rewards = deque(maxlen=100)
    intrinsic_rewards = []
    losses = []

    # Calculer le nombre total d'étapes pour l'exploration
    total_steps = Config.NUM_EPISODES * 1000  # Estimation du nombre d'étapes par épisode
    exploration_steps = int(total_steps * Config.EXPLORATION_FRACTION)

    # Boucle d'entraînement
    for episode in range(Config.NUM_EPISODES):
        # Réinitialisation de l'environnement
        state = env.reset()
        episode_reward = 0
        episode_intrinsic_reward = 0
        episode_start_time = time.time()
        episode_loss = []
        episode_steps = 0

        # Boucle d'un épisode
        while True:
            # Mise à jour d'epsilon avec une décroissance linéaire
            if agent.total_steps < exploration_steps:
                agent.epsilon = Config.EPSILON_START - agent.total_steps * (
                    Config.EPSILON_START - Config.FINAL_EXPLORATION) / exploration_steps
            else:
                agent.epsilon = Config.FINAL_EXPLORATION

            # Sélection de l'action (epsilon-greedy)
            action = agent.select_action(state)

            # Exécution de l'action
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_steps += 1

            # Stockage de l'expérience dans le buffer avec récompense intrinsèque
            intrinsic_reward = agent.store_transition(state, action, reward, next_state, done)
            episode_intrinsic_reward += intrinsic_reward

            # Passage à l'état suivant
            state = next_state

            # Apprentissage
            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)

            # Fin de l'épisode
            if done:
                break

        # Mise à jour du réseau cible
        if episode % Config.TARGET_UPDATE == 0:
            agent.update_target_network()

        # Calcul du temps d'épisode
        episode_duration = time.time() - episode_start_time

        # Enregistrement des récompenses
        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)
        intrinsic_rewards.append(episode_intrinsic_reward)

        # Enregistrement de la perte moyenne
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        losses.append(avg_loss)

        # Affichage des statistiques
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        print(f"Episode {episode}, Reward: {episode_reward:.2f}, "
              f"Avg Reward (100): {avg_reward:.2f}, "
              f"Intrinsic: {episode_intrinsic_reward:.2f}, "
              f"Epsilon: {agent.epsilon:.4f}, "
              f"Loss: {avg_loss:.4f}, "
              f"Duration: {episode_duration:.2f}s, "
              f"Steps: {episode_steps}")

        # Sauvegarde périodique du modèle et visualisation
        if episode % Config.SAVE_INTERVAL == 0:
            # Créer le répertoire de sauvegarde
            os.makedirs(Config.SAVE_DIR, exist_ok=True)

            # Sauvegarder le modèle
            metrics = {
                "rewards": episode_rewards,
                "intrinsic_rewards": intrinsic_rewards,
                "losses": losses
            }
            agent.save(f"{Config.SAVE_DIR}/montezuma_dqn_ep{episode}.pth", episode, metrics)
            print(f"Model saved at episode {episode}")

            # Visualiser les récompenses
            plt.figure(figsize=(15, 10))

            plt.subplot(2, 2, 1)
            plt.plot(episode_rewards)
            plt.title("Episode Rewards")
            plt.xlabel("Episode")
            plt.ylabel("Reward")

            plt.subplot(2, 2, 2)
            plt.plot(intrinsic_rewards)
            plt.title("Intrinsic Rewards")
            plt.xlabel("Episode")
            plt.ylabel("Intrinsic Reward")

            plt.subplot(2, 2, 3)
            # Calculer la moyenne mobile sur 100 épisodes
            if len(episode_rewards) >= 100:
                moving_avg = [np.mean(episode_rewards[max(0, i-99):i+1]) for i in range(len(episode_rewards))]
                plt.plot(moving_avg)
                plt.title("100-Episode Moving Average Reward")
                plt.xlabel("Episode")
                plt.ylabel("Average Reward")

            plt.subplot(2, 2, 4)
            plt.plot(losses)
            plt.title("Training Loss")
            plt.xlabel("Episode")
            plt.ylabel("Loss")

            plt.tight_layout()
            plt.savefig(f"{Config.SAVE_DIR}/metrics_ep{episode}.png")
            plt.close()

    # Fermeture de l'environnement
    env.close()

    return agent.policy_net