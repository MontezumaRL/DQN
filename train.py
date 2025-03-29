# fichier: train.py
import torch
import numpy as np
import time
from collections import deque
import random

# Utilise wandb pour le logging (optionnel mais recommandé)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("wandb not installed, logging to console only.")
    WANDB_AVAILABLE = False


from src.environment import MontezumaEnvironment
from src.agent import DQNAgent
from src import config as cfg # Importer la configuration

def main(resume_training=False):
    # Initialisation de wandb (si disponible)
    if WANDB_AVAILABLE:
        wandb.init(
            project="dqn-rnd-montezuma",
            config={
                "learning_rate_dqn": cfg.LEARNING_RATE_DQN,
                "learning_rate_rnd": cfg.RND_LR if cfg.USE_RND else None,
                "batch_size": cfg.BATCH_SIZE,
                "buffer_size": cfg.BUFFER_SIZE,
                "gamma": cfg.GAMMA,
                "target_update_freq": cfg.TARGET_UPDATE_FREQ,
                "epsilon_decay_frames": cfg.EPSILON_DECAY_FRAMES,
                "use_rnd": cfg.USE_RND,
                "intrinsic_reward_scale": cfg.INTRINSIC_REWARD_SCALE if cfg.USE_RND else None,
                "seed": cfg.SEED,
            },
            resume="allow", # Permet de reprendre un run existant
            id=f"montezuma-ddqn-rnd-{cfg.SEED}" # Donne un ID unique basé sur le seed
        )

    print(f"Using device: {cfg.DEVICE}")
    print(f"Seed: {cfg.SEED}")
    print(f"Using RND: {cfg.USE_RND}")

    # --- Initialisation ---
    # Pour la reproductibilité
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)
        # Note: Les opérations CUDA non-déterministes peuvent subsister
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False # Peut ralentir, mais améliore la reproductibilité

    env = MontezumaEnvironment(render_mode=None, seed=cfg.SEED) # Pas de rendu pendant l'entraînement rapide
    n_actions = env.action_space.n

    agent = DQNAgent(env, n_actions)

    if resume_training:
        agent.load_models()
        # Note: Si tu veux reprendre epsilon/total_steps, charge-les ici
        # Par exemple, charge un fichier 'training_state.pkl' qui les contient

    # --- Variables de suivi ---
    episode_rewards = deque(maxlen=100) # Récompenses totales (ext + int) des 100 derniers épisodes
    episode_extrinsic_rewards = deque(maxlen=100) # Récompenses extrinsèques seules
    episode_intrinsic_rewards = deque(maxlen=100) # Récompenses intrinsèques seules
    episode_lengths = deque(maxlen=100)

    total_frames = agent.total_steps # Reprend le compte si chargé
    start_time = time.time()
    episode = 0

    print("Starting training...")
    state, info = env.reset(seed=cfg.SEED + episode) # Seed différent pour chaque épisode
    current_episode_reward = 0.0
    current_episode_extrinsic_reward = 0.0
    current_episode_intrinsic_reward = 0.0
    current_episode_length = 0

    while total_frames < cfg.NUM_FRAMES:

        # --- Interaction avec l'environnement ---
        action = agent.choose_action(state) # total_steps est incrémenté ici
        next_state, extrinsic_reward, done, info = env.step(action)

        # --- Stockage de la transition (calcule RND si besoin) ---
        # L'agent calcule la récompense RND normalisée et la combine
        agent.store_transition(state, action, extrinsic_reward, next_state, done)

        # Récupère la récompense intrinsèque pour le logging (avant scaling)
        # Note: On recalcule juste pour le log, ce n'est pas ce qui est stocké
        # Si tu veux la vraie valeur stockée, il faudrait modifier store_transition
        intrinsic_reward_for_log = 0.0
        if cfg.USE_RND and agent.memory: # Vérifie que memory n'est pas vide
           last_transition = agent.memory.buffer[-1]
           # La récompense stockée est déjà r_ext + scale * r_int_norm
           # On approxime r_int_norm = (total_reward - r_ext) / scale
           total_reward_stored = last_transition[2]
           extrinsic_reward_stored = extrinsic_reward # Approximatif si délai
           if cfg.INTRINSIC_REWARD_SCALE > 1e-6:
                intrinsic_reward_for_log = (total_reward_stored - extrinsic_reward_stored) / cfg.INTRINSIC_REWARD_SCALE


        # --- Mise à jour des compteurs ---
        state = next_state
        current_episode_reward += extrinsic_reward + cfg.INTRINSIC_REWARD_SCALE * intrinsic_reward_for_log # Utilise la valeur loggée
        current_episode_extrinsic_reward += extrinsic_reward
        current_episode_intrinsic_reward += intrinsic_reward_for_log # Intrinsèque normalisé (non-scalé)
        current_episode_length += 1
        total_frames += 1 # Attention: agent.total_steps est déjà incrémenté dans choose_action

        # --- Entraînement des réseaux ---
        dqn_loss, rnd_loss = agent.update_networks() # Fait la mise à jour si buffer assez rempli

        # --- Logging ---
        if total_frames % cfg.LOG_FREQ == 0 and dqn_loss is not None:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            avg_ext_reward = np.mean(episode_extrinsic_rewards) if episode_extrinsic_rewards else 0.0
            avg_int_reward = np.mean(episode_intrinsic_rewards) if episode_intrinsic_rewards else 0.0
            avg_length = np.mean(episode_lengths) if episode_lengths else 0.0
            elapsed_time = time.time() - start_time
            fps = cfg.LOG_FREQ / elapsed_time if elapsed_time > 0 else 0

            print(f"Frames: {total_frames}/{cfg.NUM_FRAMES} | FPS: {fps:.2f} | Epsilon: {agent.epsilon:.4f} | "
                  f"Avg Reward (100 ep): {avg_reward:.2f} | Avg Ext Reward: {avg_ext_reward:.2f} | "
                  f"Avg Int Reward: {avg_int_reward:.4f} | Avg Length: {avg_length:.1f} | "
                  f"DQN Loss: {dqn_loss:.4f}" + (f" | RND Loss: {rnd_loss:.4f}" if rnd_loss is not None else ""))

            if WANDB_AVAILABLE:
                 log_data = {
                     "train/epsilon": agent.epsilon,
                     "train/dqn_loss": dqn_loss,
                     "reward/avg_100ep_total_reward": avg_reward,
                     "reward/avg_100ep_extrinsic_reward": avg_ext_reward,
                     "env/avg_100ep_length": avg_length,
                     "perf/fps": fps,
                     "perf/total_frames": total_frames,
                 }
                 if rnd_loss is not None:
                     log_data["train/rnd_loss"] = rnd_loss
                     log_data["reward/avg_100ep_intrinsic_reward"] = avg_int_reward
                 # Log des stats de normalisation RND (si utilisées)
                 if cfg.USE_RND:
                    log_data["rnd/obs_norm_mean"] = agent.obs_normalizer.rms.mean.mean() # Moyenne globale de la moyenne des obs
                    log_data["rnd/obs_norm_std"] = agent.obs_normalizer.rms.std.mean()   # Moyenne globale du std des obs
                    log_data["rnd/int_reward_norm_std"] = agent.intrinsic_reward_rms.std

                 wandb.log(log_data, step=total_frames)

            start_time = time.time() # Reset timer pour le prochain log

        # --- Sauvegarde du modèle ---
        if total_frames % cfg.SAVE_FREQ == 0:
            agent.save_models()
            # Sauvegarder l'état de l'entraînement (total_frames, epsilon) si besoin
            # with open('training_state.pkl', 'wb') as f:
            #    pickle.dump({'total_frames': total_frames, 'epsilon': agent.epsilon}, f)


        # --- Fin de l'épisode ---
        if done:
            episode += 1
            episode_rewards.append(current_episode_reward)
            episode_extrinsic_rewards.append(current_episode_extrinsic_reward)
            episode_intrinsic_rewards.append(current_episode_intrinsic_reward)
            episode_lengths.append(current_episode_length)

            if WANDB_AVAILABLE:
                wandb.log({
                    "reward/episode_total_reward": current_episode_reward,
                    "reward/episode_extrinsic_reward": current_episode_extrinsic_reward,
                    "reward/episode_intrinsic_reward": current_episode_intrinsic_reward,
                    "env/episode_length": current_episode_length,
                    "env/episode_count": episode,
                }, step=total_frames)

            print(f"Episode {episode} finished after {current_episode_length} steps. Total Reward: {current_episode_reward:.2f} "
                  f"(Ext: {current_episode_extrinsic_reward:.2f}, Int: {current_episode_intrinsic_reward:.4f}). Frames: {total_frames}")

            # Reset pour le nouvel épisode
            state, info = env.reset(seed=cfg.SEED + episode)
            current_episode_reward = 0.0
            current_episode_extrinsic_reward = 0.0
            current_episode_intrinsic_reward = 0.0
            current_episode_length = 0

    # --- Fin de l'entraînement ---
    print("Training finished.")
    agent.save_models() # Sauvegarde finale
    env.close()
    if WANDB_AVAILABLE:
        wandb.finish()

if __name__ == "__main__":
    # Ajoute un argument pour reprendre l'entraînement si tu veux
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training from saved models")
    args = parser.parse_args()

    main(resume_training=args.resume)