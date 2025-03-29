# fichier: evaluate.py
import torch
import time
import argparse
import numpy as np

from src.environment import MontezumaEnvironment
from src.networks import DQN_CNN # On a besoin de l'architecture pour charger les poids
from src import config as cfg

def evaluate_agent(model_path, num_episodes=10, render_mode='human', seed=None):
    """
    Évalue un agent DQN entraîné sur l'environnement MontezumaRevenge.

    Args:
        model_path (str): Chemin vers le fichier de poids du modèle (.pth).
        num_episodes (int): Nombre d'épisodes à exécuter pour l'évaluation.
        render_mode (str): Mode de rendu de Gymnasium ('human', 'rgb_array', None).
        seed (int, optional): Graine pour la reproductibilité de l'environnement.
    """

    print(f"Using device: {cfg.DEVICE}")
    print(f"Loading model from: {model_path}")
    print(f"Running {num_episodes} episodes with render_mode='{render_mode}'")
    if seed is not None:
        print(f"Using seed: {seed}")

    # --- Initialisation ---
    env = MontezumaEnvironment(render_mode=render_mode, seed=seed)
    n_actions = env.action_space.n

    # Charger uniquement le réseau de politique
    policy_net = DQN_CNN(cfg.INPUT_SHAPE, n_actions).to(cfg.DEVICE)

    try:
        # Charger les poids sauvegardés
        policy_net.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        env.close()
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        env.close()
        return

    # Mettre le réseau en mode évaluation (très important !)
    # Désactive le dropout, met à jour les stats de batch norm différemment, etc.
    policy_net.eval()

    episode_rewards = []
    episode_lengths = []

    # --- Boucle d'évaluation ---
    for i_episode in range(num_episodes):
        print(f"\nStarting Episode {i_episode + 1}/{num_episodes}")
        # Utiliser une seed différente pour chaque épisode si une seed de base est fournie
        current_seed = seed + i_episode if seed is not None else None
        state, info = env.reset(seed=current_seed)
        done = False
        episode_reward = 0
        episode_length = 0
        start_time = time.time()

        while not done:
            # Convertir l'état en tenseur pour le réseau
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(cfg.DEVICE)

            # Choisir l'action de manière gourmande (pas d'epsilon-exploration)
            with torch.no_grad(): # Désactive le calcul des gradients (plus rapide et moins de mémoire)
                q_values = policy_net(state_tensor)
                action = q_values.argmax(dim=1).item() # Choisir l'action avec la Q-value max

            # Exécuter l'action dans l'environnement
            next_state, reward, done, info = env.step(action)

            # Mettre à jour l'état, la récompense et la longueur
            state = next_state
            episode_reward += reward # Utilise la récompense extrinsèque de l'env
            episode_length += 1

            # Ralentir un peu si on rend pour pouvoir voir ce qu'il se passe
            if render_mode == 'human':
                time.sleep(0.02) # Pause de 20ms

        # Fin de l'épisode
        end_time = time.time()
        episode_duration = end_time - start_time
        print(f"Episode {i_episode + 1} finished.")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length} steps")
        print(f"  Duration: {episode_duration:.2f} seconds")
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    # --- Fin de l'évaluation ---
    env.close()
    print("\n--- Evaluation Summary ---")
    print(f"Average Reward over {num_episodes} episodes: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Average Length over {num_episodes} episodes: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent for Montezuma's Revenge.")
    parser.add_argument(
        "--model",
        type=str,
        default=cfg.MODEL_SAVE_PATH, # Utilise le chemin par défaut de config.py
        help=f"Path to the trained policy network file (.pth). Default: {cfg.MODEL_SAVE_PATH}"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run for evaluation. Default: 5"
    )
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction, # Permet --render ou --no-render
        default=True,
        help="Enable/disable rendering ('human' mode). Default: enabled (--render)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None, # Pas de seed fixe par défaut pour voir la généralisation
        help="Optional seed for the environment initialization for reproducible evaluation runs."
    )

    args = parser.parse_args()

    render_mode = 'human' if args.render else None
    evaluate_agent(model_path=args.model, num_episodes=args.episodes, render_mode=render_mode, seed=args.seed)