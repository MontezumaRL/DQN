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

        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            agent.learn()

            if episode_start_time + 10 < time.time():
                done = True

            if done:
                break

        if episode % Config.TARGET_UPDATE == 0:
            agent.update_target_network()

        agent.update_epsilon()
        episode_duration = time.time() - episode_start_time
        episode_rewards.append(episode_reward)

        print(f"Episode {episode}, Reward: {episode_reward}, "
              f"Epsilon: {agent.epsilon:.2f}, Duration: {episode_duration:.2f}s, "
              f"Steps: {agent.total_steps}")

        if episode % Config.SAVE_INTERVAL == 0:
            os.makedirs(Config.SAVE_DIR, exist_ok=True)
            # Sauvegarde avec le même format que celui utilisé dans evaluate_model
            save_path = f"{Config.SAVE_DIR}/montezuma_dqn_ep_step2{episode}.pth"
            torch.save({
                'policy_net_state_dict': agent.policy_net.state_dict(),
                'episode': episode,
                'episode_rewards': episode_rewards
            }, save_path)
            print(f"Model saved at episode {episode}")

    env.close()
    return agent.policy_net

def evaluate_model(model_path, num_episodes=5, render=True, start_x=21, start_y=192):
    """
    Évalue un modèle DQN sauvegardé.

    Args:
        model_path (str): Chemin vers le fichier du modèle sauvegardé
        num_episodes (int): Nombre d'épisodes d'évaluation
        render (bool): Si True, affiche le rendu de l'environnement
        start_x (int): Position initiale x de l'agent
        start_y (int): Position initiale y de l'agent
    """
    total_rewards = []

    try:
        # Création de l'environnement
        render_mode = "human" if render else None
        env = MontezumaEnvironment(render_mode=render_mode)

        # Détection du device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Création de l'agent
        agent = DQNAgent(
            input_shape=(4, 42, 42),
            n_actions=env.n_actions,
            device=device,
            config=Config
        )

        # Chargement du checkpoint
        checkpoint = torch.load(model_path)
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.policy_net.eval()  # Mode évaluation

        for episode in range(num_episodes):
            state = env.reset()
            env.set_agent_position(start_x, start_y)
            episode_reward = 0
            done = False

            while not done:
                # Sélection de l'action de façon déterministe
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = agent.policy_net(state_tensor).max(1)[1].item()

                # Exécution de l'action
                state, reward, done, info = env.step(action)
                episode_reward += reward

            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

        # Statistiques finales
        if total_rewards:
            avg_reward = sum(total_rewards) / len(total_rewards)
            print(f"\nÉvaluation terminée sur {num_episodes} épisodes:")
            print(f"Récompense moyenne: {avg_reward:.2f}")
            print(f"Récompense min: {min(total_rewards)}")
            print(f"Récompense max: {max(total_rewards)}")

    except Exception as e:
        print(f"Erreur lors de l'évaluation: {str(e)}")
        traceback.print_exc()  # Affiche le traceback complet

    finally:
        if 'env' in locals():
            env.close()

    return total_rewards