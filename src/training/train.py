import os
import time

import torch
from ..agents.dqn import DQNAgent
from ..config import Config
from ..environment import MontezumaEnvironment


def train_montezuma(checkpoint_path=None, start_x=21, start_y=192):
    env = MontezumaEnvironment(render_mode="human" if Config.RENDER else None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialisation de l'agent
    agent = DQNAgent((4, 42, 42), env.n_actions, device, Config)

    # Chargement du modèle pré-entraîné si spécifié
    start_episode = 0
    episode_rewards = []
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['policy_net_state_dict'])
        print(f"Loaded model successfully")

        # Réinitialisation d'epsilon à 0.99
        agent.epsilon = 0.79
        #agent.update_epsilon()
        print(f"Reset epsilon to {agent.epsilon}")

    # Boucle d'entraînement
    for episode in range(start_episode, Config.NUM_EPISODES):
        state = env.reset()
        env.set_agent_position(start_x, start_y)
        #env.setup_position_human()
        episode_reward = 0
        episode_start_time = time.time()
        nb_steps = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            nb_steps += 1

            # Stockage de l'expérience dans le buffer
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            agent.learn()

            if episode_start_time + 10 < time.time():
                done = True

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
            # Sauvegarde avec le même format que celui utilisé dans evaluate_model
            save_path = f"{Config.SAVE_DIR}/montezuma_dqn_ep_step{episode}.pth"
            torch.save({
                'policy_net_state_dict': agent.policy_net.state_dict(),
                'episode': episode,
                'episode_rewards': episode_rewards
            }, save_path)
            print(f"Model saved at episode {episode}")

    # Fermeture de l'environnement
    env.close()

    return agent.policy_net