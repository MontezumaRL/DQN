import torch
from ..environment import MontezumaEnvironment
from ..models.networks import DQNNetwork
from ..agents.dqn import DQNAgent
from ..config import Config


import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from collections import defaultdict
import gymnasium as gym
from ..environment import MontezumaEnvironment, FrameSkipWrapper
from ..agents.dqn import DQNAgent
from ..config import Config


def evaluate_model(model_path, num_episodes=5, record_video=True, render=True, frame_skip=4):
    """
    Évalue un modèle DQN sauvegardé sur plusieurs épisodes

    Args:
        model_path: Chemin vers le fichier du modèle sauvegardé
        num_episodes: Nombre d'épisodes d'évaluation
        record_video: Si True, enregistre une vidéo de l'évaluation
        render: Si True, affiche le rendu de l'environnement
        frame_skip: Nombre de frames à sauter entre chaque action

    Returns:
        dict: Statistiques d'évaluation
    """
    # Création du dossier pour les vidéos
    eval_dir = "evaluation_results"
    os.makedirs(eval_dir, exist_ok=True)

    # Extraction du numéro d'épisode du nom du fichier
    model_name = os.path.basename(model_path).split('.')[0]

    # Configuration de l'environnement
    render_mode = "human" if render else None
    env = FrameSkipWrapper(
        MontezumaEnvironment(render_mode=render_mode),
        skip=frame_skip
    )

    # Configuration de l'enregistrement vidéo si demandé
    if record_video:
        video_dir = os.path.join(eval_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"{model_name}_eval.mp4")
        env.env.env = gym.wrappers.RecordVideo(
            env.env.env,
            video_folder=video_dir,
            name_prefix=model_name,
            episode_trigger=lambda x: True  # Enregistrer tous les épisodes
        )

    # Détection du device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialisation de l'agent
    agent = DQNAgent((4, 84, 84), env.n_actions, device, Config)

    # Chargement du modèle sauvegardé
    episode, metrics = agent.load(model_path)
    print(f"Modèle chargé depuis l'épisode {episode}")

    # Passage en mode évaluation
    agent.policy_net.eval()

    # Statistiques d'évaluation
    stats = defaultdict(list)

    # Boucle d'évaluation
    for ep in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_intrinsic_reward = 0
        done = False
        steps = 0
        start_time = time.time()

        # Dictionnaire pour suivre les actions prises
        action_counts = defaultdict(int)

        # Liste pour suivre les positions visitées
        visited_positions = set()

        while not done:
            # Sélection de l'action de façon déterministe (epsilon = 0)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = agent.policy_net(state_tensor).max(1)[1].item()

            # Calculer la récompense intrinsèque (pour information seulement)
            intrinsic_reward = agent.compute_intrinsic_reward(state)
            episode_intrinsic_reward += intrinsic_reward

            # Exécution de l'action
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            steps += 1

            # Suivre les actions prises
            action_counts[action] += 1

            # Suivre les positions visitées (approximation)
            if hasattr(env.env, '_get_agent_position'):
                pos = env.env._get_agent_position(state)
                visited_positions.add(pos)

            # Limiter le nombre d'étapes pour éviter les épisodes trop longs
            if steps > 10000:
                print("Épisode trop long, arrêt forcé.")
                done = True

        # Calculer les statistiques de l'épisode
        duration = time.time() - start_time

        # Enregistrer les statistiques
        stats['rewards'].append(episode_reward)
        stats['intrinsic_rewards'].append(episode_intrinsic_reward)
        stats['steps'].append(steps)
        stats['durations'].append(duration)
        stats['unique_positions'].append(len(visited_positions))

        # Calculer la distribution des actions
        action_distribution = {f"action_{a}": count/steps for a, count in action_counts.items()}
        for action, freq in action_distribution.items():
            stats[action].append(freq)

        print(f"Épisode {ep+1}/{num_episodes}, Récompense: {episode_reward:.2f}, "
              f"Intrinsèque: {episode_intrinsic_reward:.2f}, Étapes: {steps}, "
              f"Durée: {duration:.2f}s, Positions uniques: {len(visited_positions)}")

    # Calculer les moyennes
    avg_reward = np.mean(stats['rewards'])
    avg_intrinsic = np.mean(stats['intrinsic_rewards'])
    avg_steps = np.mean(stats['steps'])
    avg_positions = np.mean(stats['unique_positions'])

    print(f"\nRésultats sur {num_episodes} épisodes:")
    print(f"Récompense moyenne: {avg_reward:.2f} ± {np.std(stats['rewards']):.2f}")
    print(f"Récompense intrinsèque moyenne: {avg_intrinsic:.2f}")
    print(f"Nombre d'étapes moyen: {avg_steps:.1f}")
    print(f"Nombre moyen de positions uniques: {avg_positions:.1f}")

    # Visualiser les résultats
    plt.figure(figsize=(15, 10))

    # Graphique des récompenses
    plt.subplot(2, 2, 1)
    plt.bar(range(num_episodes), stats['rewards'], color='blue')
    plt.axhline(y=avg_reward, color='r', linestyle='-', label=f'Moyenne: {avg_reward:.2f}')
    plt.title('Récompenses par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Récompense')
    plt.legend()

    # Graphique des récompenses intrinsèques
    plt.subplot(2, 2, 2)
    plt.bar(range(num_episodes), stats['intrinsic_rewards'], color='green')
    plt.axhline(y=avg_intrinsic, color='r', linestyle='-', label=f'Moyenne: {avg_intrinsic:.2f}')
    plt.title('Récompenses intrinsèques par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Récompense intrinsèque')
    plt.legend()

    # Graphique du nombre d'étapes
    plt.subplot(2, 2, 3)
    plt.bar(range(num_episodes), stats['steps'], color='orange')
    plt.axhline(y=avg_steps, color='r', linestyle='-', label=f'Moyenne: {avg_steps:.1f}')
    plt.title('Nombre d\'étapes par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Étapes')
    plt.legend()

    # Graphique de la distribution des actions (moyenne sur tous les épisodes)
    plt.subplot(2, 2, 4)
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