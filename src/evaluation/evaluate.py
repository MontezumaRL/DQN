import torch
from ..environment import MontezumaEnvironment
from ..models.networks import DQNNetwork
from ..agents.dqn import DQNAgent
from ..config import Config


def evaluate_model(model_path, num_episodes=5):
    """
    Évalue un modèle DQN sauvegardé sur plusieurs épisodes

    Args:
        model_path: Chemin vers le fichier du modèle sauvegardé
        num_episodes: Nombre d'épisodes d'évaluation
    """
    # Création de l'environnement
    env = MontezumaEnvironment(render_mode="human")

    # Détection du device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialisation de l'agent
    agent = DQNAgent((4, 84, 84), env.n_actions, device, Config)

    # Chargement du modèle sauvegardé
    episode, _ = agent.load(model_path)
    print(f"Modèle chargé depuis l'épisode {episode}")

    # Passage en mode évaluation
    agent.policy_net.eval()

    total_rewards = []

    # Boucle d'évaluation
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        while not done:
            # Sélection de l'action de façon déterministe (epsilon = 0)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = agent.policy_net(state_tensor).max(1)[1].item()

            # Exécution de l'action
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            steps += 1

        total_rewards.append(episode_reward)
        print(f"Épisode {episode}, Récompense: {episode_reward}, Étapes: {steps}")

    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"Récompense moyenne sur {num_episodes} épisodes: {avg_reward:.2f}")

    env.close()

    return total_rewards