import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from collections import defaultdict
import gymnasium as gym
from ..environment import MontezumaEnvironment
from ..agents.ddqn import DuelingDQNAgent  # Changé pour DuelingDQNAgent
from ..config_ddqn import DuelingDQNConfig  # Changé pour DuelingDQNConfig

def evaluate_model(model_path, num_episodes=5, start_x=21, start_y=192):
    """
    Évalue un modèle Dueling DQN sauvegardé sur plusieurs épisodes

    Args:
        model_path: Chemin vers le fichier du modèle sauvegardé
        num_episodes: Nombre d'épisodes d'évaluation
        start_x, start_y: Position de départ de l'agent

    Returns:
        dict: Statistiques d'évaluation
    """
    # Création du dossier pour les vidéos
    eval_dir = "output/evaluation"
    os.makedirs(eval_dir, exist_ok=True)

    # Extraction du numéro d'épisode du nom du fichier
    model_name = os.path.basename(model_path).split('.')[0]

    # Configuration de l'environnement
    env = MontezumaEnvironment(render_mode="human" if DuelingDQNConfig.RENDER else None)

    # Détection du device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialisation de l'agent
    agent = DuelingDQNAgent((4, 42, 42), env.n_actions, device, DuelingDQNConfig)

    # Chargement du checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        print("Modèle chargé avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return None

    # Passage en mode évaluation
    agent.policy_net.eval()

    # Statistiques d'évaluation
    stats = defaultdict(list)

    # Boucle d'évaluation
    for ep in range(num_episodes):
        state = env.reset()
        env.set_agent_position(start_x, start_y)
        episode_reward = 0
        done = False
        steps = 0
        start_time = time.time()

        # Dictionnaire pour suivre les actions prises
        action_counts = defaultdict(int)

        # Liste pour suivre les positions visitées
        visited_positions = set()

        while not done:
            # Sélection de l'action de façon déterministe
            action = agent.select_action(state, training=False)

            # Exécution de l'action
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            steps += 1

            # Suivre les actions prises
            action_counts[action] += 1

            # Limiter le nombre d'étapes
            if steps > 10000:
                print("Épisode trop long, arrêt forcé.")
                done = True

        # Calculer les statistiques de l'épisode
        duration = time.time() - start_time

        # Enregistrer les statistiques
        stats['rewards'].append(episode_reward)
        stats['steps'].append(steps)
        stats['durations'].append(duration)

        # Calculer la distribution des actions
        action_distribution = {f"action_{a}": count/steps for a, count in action_counts.items()}
        for action, freq in action_distribution.items():
            stats[action].append(freq)

        print(f"Épisode {ep+1}/{num_episodes}, Récompense: {episode_reward:.2f}, "
              f"Pas: {steps}, Durée: {duration:.2f}s")

    # Calculer et afficher les statistiques
    avg_reward = np.mean(stats['rewards'])
    avg_steps = np.mean(stats['steps'])

    print(f"\nRésultats sur {num_episodes} épisodes:")
    print(f"Récompense moyenne: {avg_reward:.2f} ± {np.std(stats['rewards']):.2f}")
    print(f"Nombre de pas moyen: {avg_steps:.1f}")

    # Créer les visualisations
    plt.figure(figsize=(15, 10))

    # Graphique des récompenses
    plt.subplot(2, 2, 1)
    plt.bar(range(num_episodes), stats['rewards'], color='blue')
    plt.axhline(y=avg_reward, color='r', linestyle='-', label=f'Moyenne: {avg_reward:.2f}')
    plt.title('Récompenses par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Récompense')
    plt.legend()

    # Graphique du nombre d'étapes
    plt.subplot(2, 2, 2)
    plt.bar(range(num_episodes), stats['steps'], color='orange')
    plt.axhline(y=avg_steps, color='r', linestyle='-', label=f'Moyenne: {avg_steps:.1f}')
    plt.title('Nombre d\'étapes par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Pas')
    plt.legend()

    # Graphique de la distribution des actions
    plt.subplot(2, 2, 3)
    action_keys = [k for k in stats.keys() if k.startswith('action_')]
    if action_keys:
        action_avgs = [np.mean(stats[k]) for k in action_keys]
        action_labels = [k.replace('action_', '') for k in action_keys]
        plt.bar(action_labels, action_avgs, color='purple')
        plt.title('Distribution moyenne des actions')
        plt.xlabel('Action')
        plt.ylabel('Fréquence')
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'Pas de données d\'actions disponibles',
                 horizontalalignment='center', verticalalignment='center')

    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, f"{model_name}_evaluation.png"))
    print(f"Graphique d'évaluation sauvegardé dans {eval_dir}/{model_name}_evaluation.png")

    # Fermer l'environnement
    env.close()

    return stats