import os
import time
from typing import Optional
import torch
import numpy as np
from ..agents.ddqn import DuelingDQNAgent
from ..config_ddqn import DuelingDQNConfig
from ..environment import MontezumaEnvironment
from ..memory.replay_buffer import ReplayBuffer

def train_montezuma_dueling(
    checkpoint_path: Optional[str] = None,
    start_x: int = 21,
    start_y: int = 192
) -> torch.nn.Module:
    # Configuration de l'environnement et du device
    env = MontezumaEnvironment(
        render_mode="human" if DuelingDQNConfig.RENDER else None
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialisation de l'agent et du buffer
    agent = DuelingDQNAgent(
        input_shape=(4, 42, 42),
        n_actions=env.n_actions,
        device=device,
        config=DuelingDQNConfig
    )
    replay_buffer = ReplayBuffer(DuelingDQNConfig.REPLAY_SIZE)

    # Métriques de suivi
    start_episode = 0
    episode_rewards = []
    best_reward = float('-inf')
    running_reward = 0
    training_start_time = time.time()

    # Chargement du checkpoint si spécifié
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint.get('episode', 0)
            episode_rewards = checkpoint.get('episode_rewards', [])
            print(f"Loaded checkpoint from episode {start_episode}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    try:
        for episode in range(start_episode, DuelingDQNConfig.NUM_EPISODES):
            state = env.reset()
            env.set_agent_position(start_x, start_y)
            episode_reward = 0
            episode_losses = []
            episode_start_time = time.time()

            # Boucle d'épisode
            for step in range(DuelingDQNConfig.MAX_STEPS):
                # Sélection et exécution de l'action
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)

                # Stockage direct dans le buffer
                replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward

                # Apprentissage
                if len(replay_buffer) > DuelingDQNConfig.REPLAY_INITIAL:
                    if step % DuelingDQNConfig.LEARN_EVERY == 0:
                        experiences = replay_buffer.sample(DuelingDQNConfig.BATCH_SIZE)
                        loss = agent.learn(experiences)
                        if loss is not None:
                            episode_losses.append(loss)

                    if step % DuelingDQNConfig.UPDATE_TARGET_EVERY == 0:
                        agent.update_target_network()

                # Vérification du timeout
                if time.time() - episode_start_time > DuelingDQNConfig.MAX_EPISODE_DURATION:
                    print("Episode timeout")
                    done = True

                if done:
                    break

            # Mise à jour des métriques
            episode_duration = time.time() - episode_start_time
            episode_rewards.append(episode_reward)
            running_reward = 0.05 * episode_reward + 0.95 * running_reward
            mean_loss = np.mean(episode_losses) if episode_losses else 0

            # Mise à jour d'epsilon
            agent.update_epsilon()

            # Sauvegarde du meilleur modèle
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save(f"{DuelingDQNConfig.SAVE_DIR}/best_model.pth")

            # Sauvegarde périodique
            if episode % DuelingDQNConfig.SAVE_INTERVAL == 0:
                save_path = f"{DuelingDQNConfig.SAVE_DIR}/checkpoint_ep{episode}.pth"
                agent.save(save_path, {
                    'episode': episode,
                    'episode_rewards': episode_rewards,
                    'best_reward': best_reward,
                    'running_reward': running_reward
                })

            # Affichage des statistiques tous les 500 épisodes
            if episode % 100 == 0:
                print(
                    f"Episode {episode}/{DuelingDQNConfig.NUM_EPISODES} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Running reward: {running_reward:.2f} | "
                    f"Epsilon: {agent.epsilon:.3f} | "
                    f"Loss: {mean_loss:.4f} | "
                    f"Duration: {episode_duration:.2f}s | "
                    f"Buffer: {len(replay_buffer)}/{DuelingDQNConfig.REPLAY_SIZE}"
                )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        # Sauvegarde finale
        agent.save(f"{DuelingDQNConfig.SAVE_DIR}/final_model.pth")
        env.close()

        # Statistiques finales
        total_duration = time.time() - training_start_time
        print(f"\nTraining completed in {total_duration/3600:.2f} hours")
        print(f"Best reward achieved: {best_reward}")
        print(f"Final running reward: {running_reward}")

    return agent.policy_net