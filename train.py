import torch
import numpy as np
import time
from collections import deque
import random
import argparse # Importer argparse ici
#from gymnasium.wrappers import TimeLimit

# --- Ajouts pour le Profiling ---
import cProfile
import pstats
import io # Pour capturer la sortie de pstats dans une chaîne
# ----------------------------------

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

# --- Fonction contenant la boucle à profiler ---
# --- Fonction contenant la boucle à profiler ---
def run_training_steps(agent, env, start_episode, profile_num_frames, initial_info):
    """
    Exécute un nombre défini de frames d'entraînement sous le contrôle du profiler.
    Cette fonction contient la logique principale qui sera profilée.

    Args:
        agent (DQNAgent): L'agent RL.
        env (MontezumaEnvironment): L'environnement.
        start_episode (int): Le numéro de l'épisode de départ pour ce run.
        profile_num_frames (int): Nombre de frames à exécuter.
        initial_info (dict): Le dictionnaire info retourné par le premier env.reset().

    Returns:
        int: Le numéro de l'épisode atteint à la fin du profiling.
        dict: Le dernier dictionnaire info retourné par env.reset().
    """
    episode = start_episode
    start_time_profiling = time.time() # Timer pour la durée du profiling
    initial_steps = agent.total_steps # Nombre de pas au début du profiling
    target_steps = initial_steps + profile_num_frames

    print(f"--- Starting Profiling Run ---")
    print(f"Initial steps: {initial_steps}")
    print(f"Target steps: {target_steps}")
    print(f"Profiling for {profile_num_frames} frames...")

    # Récupérer l'état initial du reset fait avant d'appeler cette fonction
    state = initial_info['initial_state'] # Assurez-vous que l'état initial est passé via info
    current_start_location = initial_info.get('start_location', 'unknown_profile_start')
    info = initial_info # Garder le reste des infos

    # Utiliser les deques globales pour le log pendant le profiling si besoin
    global episode_rewards, episode_extrinsic_rewards, episode_intrinsic_rewards, episode_lengths, episode_start_locations

    # --- Boucle d'entraînement principale (limitée par profile_num_frames) ---
    while agent.total_steps < target_steps:

        # Initialisation pour cet épisode
        current_episode_reward = 0.0
        current_episode_extrinsic_reward = 0.0
        current_episode_intrinsic_reward = 0.0
        current_episode_length = 0
        done = False
        episode_start_time = time.time() # Timer pour la durée de l'épisode

        # --- Boucle interne (exécution des pas de cet épisode) ---
        while not done and agent.total_steps < target_steps:

            # --- Interaction Agent-Environnement ---
            action = agent.choose_action(state) # Incrémente agent.total_steps

            # Exécuter l'action dans l'environnement
            next_state, extrinsic_reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated # Calculer done pour la condition de boucle

            # --- Stockage de la transition dans le buffer ---
            # !!! POINT CRUCIAL POUR LA MÉMOIRE !!!
            # Vérifiez que les 'state' et 'next_state' (qui sont des np.ndarray ici)
            # sont bien stockés de manière efficace dans votre ReplayBuffer.
            # Si vous les convertissiez en Tensors PyTorch AVANT de les stocker,
            # assurez-vous qu'ils sont .detach().cpu() pour éviter de garder
            # l'historique de calcul ou de les laisser sur le GPU dans un buffer CPU.
            # Avec des np.ndarray, le risque est moindre, mais vérifiez la taille du buffer.
            agent.store_transition(state, action, extrinsic_reward, next_state, done)

            # --- Calcul Récompense Intrinsèque (pour le log uniquement) ---
            # Note: Ce calcul est approximatif car il se base sur la dernière transition ajoutée
            intrinsic_reward_for_log = 0.0
            if cfg.USE_RND and len(agent.memory.buffer) > 0:
               # Attention: Accéder directement au buffer peut être lent ou non représentatif
               # Il serait préférable que store_transition retourne la récompense RND calculée
               # Pour le profilage, on peut simplifier ou ignorer ce log détaillé.
               # Calcul simplifié (peut être imprécis) :
               if abs(cfg.INTRINSIC_REWARD_SCALE) > 1e-6 and 'total_reward_stored' in agent.memory.buffer[-1]: # Si l'agent stocke la récompense totale
                    total_reward_stored = agent.memory.buffer[-1]['total_reward_stored'] # Adaptez à la structure de votre buffer
                    intrinsic_reward_for_log = (total_reward_stored - extrinsic_reward) / cfg.INTRINSIC_REWARD_SCALE

            # --- Mise à jour de l'état et des compteurs d'épisode ---
            state = next_state # L'état pour le prochain pas
            # Note: Ne pas accumuler la récompense totale ici si elle est déjà dans le buffer
            current_episode_extrinsic_reward += extrinsic_reward
            # current_episode_intrinsic_reward += intrinsic_reward_for_log # Peut être imprécis
            current_episode_length += 1

            # --- Entraînement des réseaux ---
            # C'est une partie majeure à profiler
            dqn_loss, rnd_loss = agent.update_networks()

            # --- Logging Périodique (minimum pendant profiling) ---
            if agent.total_steps % 500 == 0: # Log très peu fréquent
                 print(f"  Profiling Step: {agent.total_steps}/{target_steps} | Ep: {episode+1} | Ep Step: {current_episode_length} | Eps: {agent.epsilon:.3f}")
                 # Optionnel: Ajouter des vérifications mémoire ici si nécessaire

            # --- Vérifier si la limite de frames du profiling est atteinte ---
            if agent.total_steps >= target_steps:
                print(f"Reached profile frame limit ({target_steps}) during episode {episode + 1}.")
                break # Sortir de la boucle interne d'épisode

        # --- Fin de l'épisode ou atteinte de la limite de frames ---
        # Calculer la récompense totale de l'épisode à partir des récompenses extrinsèques/intrinsèques
        # (Ici on log juste l'extrinsèque pour la simplicité du profiling)
        episode_extrinsic_rewards.append(current_episode_extrinsic_reward)
        episode_lengths.append(current_episode_length)
        episode_start_locations.append(current_start_location) # Log d'où on a commencé

        if done:
            print(f"  Episode {episode + 1} finished naturally during profiling. Len: {current_episode_length}, ExtRew: {current_episode_extrinsic_reward:.2f}. TotFrames: {agent.total_steps}")
        else: # Sorti car limite de frames atteinte
             print(f"  Profiling stopped mid-episode {episode + 1} at step {current_episode_length}. ExtRew so far: {current_episode_extrinsic_reward:.2f}. TotFrames: {agent.total_steps}")


        # Préparer pour le prochain épisode SEULEMENT SI on n'a pas atteint la limite de frames
        if agent.total_steps < target_steps:
             episode += 1
             current_seed = cfg.SEED + episode
             state, info = env.reset(seed=current_seed) # Reset pour le prochain épisode
             current_start_location = info.get('start_location', 'unknown_profile_reset')
             print(f"  Resetting for next episode {episode + 1} during profiling (Loc: {current_start_location})...")
        else:
             print(f"  Target steps ({target_steps}) reached. Finishing profiling run.")

    profiling_duration = time.time() - start_time_profiling
    print(f"--- Finished Profiling Run ({profiling_duration:.2f} seconds) ---")
    print(f"Ending total steps: {agent.total_steps}")
    # Retourner le numéro d'épisode final et les dernières infos pour potentiellement continuer
    return episode, info


# --- Variables globales pour les logs entre épisodes ---
# (Déclarées globales pour être accessibles par run_training_steps)
episode_rewards = deque(maxlen=100)
episode_extrinsic_rewards = deque(maxlen=100)
episode_intrinsic_rewards = deque(maxlen=100)
episode_lengths = deque(maxlen=100)
# NOUVEAU: Pour log par location de départ
episode_start_locations = deque(maxlen=100)

def main(resume_training=False, profile_mode=False):
    global episode_rewards, episode_extrinsic_rewards, episode_intrinsic_rewards, episode_lengths, episode_start_locations

    # --- Configuration du Profiling ---
    PROFILE_NUM_FRAMES = 5000  # Réduit pour un test rapide, augmentez selon besoin (ex: 20k, 50k)
    PROFILER_OUTPUT_FILE = 'training_profile.prof'
    # ---------------------------------

    # --- Initialisation WandB (si disponible et non en mode profilage) ---
    run_id = f"montezuma-ddqn-rnd-{cfg.SEED}-curriculum" # Ajouter curriculum au nom
    if WANDB_AVAILABLE and not profile_mode:
        wandb.init(
            project="dqn-rnd-montezuma", # Mettez votre nom de projet
            config={ # Log la config complète
                    "learning_rate_dqn": cfg.LEARNING_RATE_DQN,
                    "learning_rate_rnd": cfg.RND_LR if cfg.USE_RND else None,
                    "batch_size": cfg.BATCH_SIZE,
                    "buffer_size": cfg.BUFFER_SIZE,
                    "gamma": cfg.GAMMA,
                    "target_update_freq": cfg.TARGET_UPDATE_FREQ,
                    "epsilon_decay_frames": cfg.EPSILON_DECAY_FRAMES,
                    "epsilon_start": cfg.EPSILON_START,
                    "epsilon_final": cfg.EPSILON_FINAL,
                    "min_buffer_size": cfg.MIN_BUFFER_SIZE,
                    "use_rnd": cfg.USE_RND,
                    "intrinsic_reward_scale": cfg.INTRINSIC_REWARD_SCALE if cfg.USE_RND else None,
                    "rnd_output_dim": cfg.RND_OUTPUT_DIM if cfg.USE_RND else None,
                    "rnd_obs_clip": cfg.RND_OBS_CLIP if cfg.USE_RND else None,
                    "rnd_reward_clip": cfg.RND_REWARD_CLIP if cfg.USE_RND else None,
                    "seed": cfg.SEED,
                    "num_frames_train": cfg.NUM_FRAMES,
                    "profile_mode": profile_mode, # Log si on est en mode profilage
                    "profiled_frames": PROFILE_NUM_FRAMES if profile_mode else None,
                },
            resume="allow",
            id=run_id,
            name=run_id
        )
        print(f"WandB initialized for run ID: {run_id}")
    elif profile_mode:
        print("--- Running in Profile Mode: WandB logging is disabled. ---")
        # Modifier la config pour le log si besoin
        if WANDB_AVAILABLE: wandb.config.update({"profile_mode": True, "profiled_frames": PROFILE_NUM_FRAMES}, allow_val_change=True)


    # --- Affichage Configuration ---
    print(f"Using device: {cfg.DEVICE}")
    print(f"Seed: {cfg.SEED}")
    print(f"Using RND: {cfg.USE_RND}")
    print(f"Using Curriculum: {cfg.USE_CURRICULUM}")
    if cfg.USE_CURRICULUM:
        print(f"  Curriculum Prob: {cfg.CURRICULUM_CONFIG.get('teleport_prob', 0)}")
        print(f"  Curriculum Locations: {list(cfg.CURRICULUM_CONFIG.get('locations', {}).keys())}")
    if profile_mode:
        print(f"Profiling for {PROFILE_NUM_FRAMES} frames.")
    else:
        print(f"Training for {cfg.NUM_FRAMES} frames.")


    # --- Initialisation Reproductibilité ---
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)
        # Options pour potentiellement améliorer la perf GPU mais réduire la reprod. exacte
        # torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = False


    # --- Initialisation Environnement & Agent ---
    # S'assurer que MAX_EPISODE_STEPS et STEP_LIMIT_PENALTY sont dans cfg
    env = MontezumaEnvironment(
        render_mode=None,
        seed=cfg.SEED,
        max_episode_steps=getattr(cfg, 'MAX_EPISODE_STEPS', 2000), # Utilise getattr pour valeur par défaut
        step_limit_penalty=getattr(cfg, 'STEP_LIMIT_PENALTY', -1.0),
        curriculum_config=cfg.CURRICULUM_CONFIG if cfg.USE_CURRICULUM else None # Passer la config curriculum
    )
    n_actions = env.action_space.n
    agent = DQNAgent(env, n_actions) # total_steps et epsilon initialisés ici

    # --- Chargement si Reprise ---
    start_episode = 0
    last_info = {} # Garder les infos du dernier reset/step
    if resume_training:
        try:
            agent.load_models() # Cette méthode devrait charger steps, optim, normalizers
            print(f"Resuming training from step {agent.total_steps}")
            # Idéalement, sauvegarder/charger le numéro d'épisode aussi
            # start_episode = loaded_episode_number # À implémenter
        except Exception as e:
             print(f"Warning: Could not load models to resume: {e}. Starting training from scratch.")
             resume_training = False # Ne pas considérer comme une reprise

    print(f"Starting training run. Initial Episode: {start_episode + 1}, Initial Total Steps: {agent.total_steps}")

    # --- Reset initial pour obtenir le premier état et info ---
    current_seed = cfg.SEED + start_episode
    initial_state, initial_reset_info = env.reset(seed=current_seed)
    # Stocker l'état initial dans les infos pour le passer à run_training_steps si besoin
    initial_reset_info['initial_state'] = initial_state
    last_info = initial_reset_info # Garder les infos du reset


    # --- Exécution : Profiling ou Entraînement Normal ---
    if profile_mode:
        # Créer le profiler
        pr = cProfile.Profile()
        print("Enabling profiler...")
        pr.enable()

        # Exécuter la section de code à profiler
        # Passe les infos initiales pour que run_training_steps ait le premier état
        final_episode_profiling, last_info_profiling = run_training_steps(agent, env, start_episode, PROFILE_NUM_FRAMES, initial_reset_info)
        last_info = last_info_profiling # Mettre à jour les dernières infos connues

        # Arrêter le profiler et analyser les résultats
        print("Disabling profiler...")
        pr.disable()
        print(f"Saving profile stats to {PROFILER_OUTPUT_FILE}")
        pr.dump_stats(PROFILER_OUTPUT_FILE)

        # Afficher les stats dans la console
        print("\n--- cProfile Stats (Top 25 by Cumulative Time) ---")
        s_cum = io.StringIO()
        ps_cum = pstats.Stats(pr, stream=s_cum).sort_stats(pstats.SortKey.CUMULATIVE)
        ps_cum.print_stats(25)
        print(s_cum.getvalue())

        print("\n--- cProfile Stats (Top 25 by Total Time) ---")
        s_tot = io.StringIO()
        ps_tot = pstats.Stats(pr, stream=s_tot).sort_stats(pstats.SortKey.TIME)
        ps_tot.print_stats(25)
        print(s_tot.getvalue())

        print("--- End of Profile Stats ---")
        print("Profiling finished. Exiting.")

    else:
        # --- Mode Entraînement Normal ---
        episode = start_episode
        # Utiliser l'état et les infos du reset initial fait plus haut
        state = initial_state
        current_start_location = initial_reset_info.get('start_location', 'unknown_initial')

        # Timer global pour FPS (peut être réinitialisé périodiquement)
        log_timer_start = time.time()

        while agent.total_steps < cfg.NUM_FRAMES:
             # Initialisation pour cet épisode
             current_episode_reward = 0.0 # Récompense totale (ext + beta * int)
             current_episode_extrinsic_reward = 0.0
             current_episode_intrinsic_reward = 0.0 # Intrinsèque normalisée/clippée
             current_episode_length = 0
             done = False
             episode_start_time = time.time()

             # --- Boucle interne d'épisode ---
             while not done:
                 action = agent.choose_action(state) # Incrémente agent.total_steps
                 next_state, extrinsic_reward, terminated, truncated, step_info = env.step(action)
                 done = terminated or truncated
                 last_info = step_info # Garder les dernières infos du pas

                 # --- Stockage (crucial: l'agent calcule la récompense totale et la stocke) ---
                 # agent.store_transition gère le calcul RND et la récompense combinée
                 agent.store_transition(state, action, extrinsic_reward, next_state, done)

                 # --- Récupération de l'intrinsèque pour log (Approximation) ---
                 # Il est préférable que l'agent retourne cette info directement si possible
                 intrinsic_reward_for_log = 0.0
                 if cfg.USE_RND and len(agent.memory.buffer) > 0:
                    last_transition = agent.memory.buffer[-1]
                    total_reward_stored = last_transition[2]
                    extrinsic_reward_of_step = extrinsic_reward
                    if abs(cfg.INTRINSIC_REWARD_SCALE) > 1e-6:
                            intrinsic_reward_for_log = (total_reward_stored - extrinsic_reward_of_step) / cfg.INTRINSIC_REWARD_SCALE


                 state = next_state
                 # Accumuler les récompenses pour le log de l'épisode
                 current_episode_extrinsic_reward += extrinsic_reward
                 current_episode_intrinsic_reward += intrinsic_reward_for_log
                 # La récompense totale accumulée devrait correspondre à la somme des récompenses stockées dans le buffer
                 current_episode_reward += extrinsic_reward + cfg.INTRINSIC_REWARD_SCALE * intrinsic_reward_for_log


                 # --- Entraînement des réseaux ---
                 dqn_loss, rnd_loss = agent.update_networks()

                 current_episode_length += 1

                 # --- Logging Principal (Périodique basé sur les pas totaux) ---
                 if agent.total_steps % cfg.LOG_FREQ == 0 and dqn_loss is not None:
                     # Calculer les moyennes sur les N derniers épisodes complets
                     avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
                     avg_ext_reward = np.mean(episode_extrinsic_rewards) if episode_extrinsic_rewards else 0.0
                     avg_int_reward = np.mean(episode_intrinsic_rewards) if episode_intrinsic_rewards else 0.0
                     avg_length = np.mean(episode_lengths) if episode_lengths else 0.0
                     elapsed_time = time.time() - log_timer_start
                     fps = cfg.LOG_FREQ / elapsed_time if elapsed_time > 0 else 0

                     print(f"Frames: {agent.total_steps}/{cfg.NUM_FRAMES} | FPS: {fps:.1f} | Ep: {episode+1} | Eps: {agent.epsilon:.4f} | "
                           f"AvgRew(100): {avg_reward:.2f} (Ext: {avg_ext_reward:.2f}, Int: {avg_int_reward:.4f}) | "
                           f"AvgLen(100): {avg_length:.1f} | DQN_Loss: {dqn_loss:.4f}" +
                           (f" | RND_Loss: {rnd_loss:.4f}" if rnd_loss is not None else ""))

                     if WANDB_AVAILABLE:
                         log_data_train = {
                             "train/epsilon": agent.epsilon,
                             "train/dqn_loss": dqn_loss,
                             "reward/avg_100ep_total_reward": avg_reward,
                             "reward/avg_100ep_extrinsic_reward": avg_ext_reward,
                             "reward/avg_100ep_intrinsic_reward": avg_int_reward,
                             "env/avg_100ep_length": avg_length,
                             "perf/fps": fps,
                             "perf/total_frames": agent.total_steps,
                             "env/buffer_size": len(agent.memory),
                             **( {"train/rnd_loss": rnd_loss} if rnd_loss is not None else {} ),
                             # Log des stats de normalisation RND (prendre la moyenne si shape > 0)
                             **( {"rnd/obs_norm_mean": np.mean(agent.obs_normalizer.rms.mean)} if cfg.USE_RND else {} ),
                             **( {"rnd/obs_norm_std": np.mean(agent.obs_normalizer.rms.std)} if cfg.USE_RND else {} ),
                             **( {"rnd/int_reward_norm_std": agent.intrinsic_reward_rms.std.item() if agent.intrinsic_reward_rms.std.size == 1 else np.mean(agent.intrinsic_reward_rms.std)} if cfg.USE_RND else {} ),
                         }
                         wandb.log(log_data_train, step=agent.total_steps)

                     log_timer_start = time.time() # Réinitialiser le timer FPS pour le prochain intervalle

                 # --- Sauvegarde Périodique ---
                 if agent.total_steps % cfg.SAVE_FREQ == 0:
                     print(f"\nSaving models at step {agent.total_steps}...")
                     agent.save_models()
                     print("Models saved.\n")


                 # --- Vérifier limite globale ---
                 if agent.total_steps >= cfg.NUM_FRAMES:
                     print("Reached total frame limit during an episode.")
                     break # Sortir de la boucle interne d'épisode

             # --- Fin de l'épisode ---
             if agent.total_steps >= cfg.NUM_FRAMES:
                 print("Training stopped due to reaching total frame limit.")
                 break # Sortir de la boucle externe (while total_steps < NUM_FRAMES)

             # Ajouter les stats de l'épisode terminé aux deques
             episode_rewards.append(current_episode_reward)
             episode_extrinsic_rewards.append(current_episode_extrinsic_reward)
             episode_intrinsic_rewards.append(current_episode_intrinsic_reward)
             episode_lengths.append(current_episode_length)
             episode_start_locations.append(current_start_location) # Log d'où on a commencé

             print(f"Episode {episode + 1} finished. Len: {current_episode_length}, Start: {current_start_location}, "
                   f"TotRew: {current_episode_reward:.2f} (Ext: {current_episode_extrinsic_reward:.2f}, Int: {current_episode_intrinsic_reward:.4f}). "
                   f"Total Frames: {agent.total_steps}")

             # Log de fin d'épisode à WandB
             if WANDB_AVAILABLE:
                  log_data_ep = {
                      "reward/episode_total_reward": current_episode_reward,
                      "reward/episode_extrinsic_reward": current_episode_extrinsic_reward,
                      "reward/episode_intrinsic_reward": current_episode_intrinsic_reward,
                      "env/episode_length": current_episode_length,
                      "env/episode_count": episode + 1,
                      "env/lives_remaining": last_info.get('lives', -1),
                      "env/terminated_by_life_loss": last_info.get('terminated_by_life_loss', False),
                      "env/terminated_by_steps": last_info.get('terminated_by_steps', False),
                      "curriculum/start_location_type": current_start_location,
                      "curriculum/teleported_start": last_info.get('teleported_start', False), # Utiliser les infos du dernier reset
                      # Log spécifique par type de départ (fin d'épisode)
                      f"reward_by_start/{current_start_location}_reward": current_episode_reward,
                      f"length_by_start/{current_start_location}_length": current_episode_length
                  }
                  wandb.log(log_data_ep, step=agent.total_steps)


             # Préparer pour le prochain épisode
             episode += 1
             current_seed = cfg.SEED + episode
             # Reset pour obtenir le nouvel état et les infos de départ (y compris la location)
             state, reset_info = env.reset(seed=current_seed)
             last_info = reset_info # Mettre à jour les dernières infos connues
             current_start_location = reset_info.get('start_location', 'unknown_next')


        # --- Fin de l'entraînement Normal ---
        print(f"\nTraining finished after {agent.total_steps} frames and {episode} episodes.")
        print("Saving final models...")
        agent.save_models()
        print("Final models saved.")
        env.close()
        if WANDB_AVAILABLE:
            print("Closing WandB run.")
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DDQN+RND agent on Montezuma's Revenge.")
    parser.add_argument("--resume", action="store_true", help="Resume training from saved models and state.")
    # --- Ajout de l'argument pour activer le mode profiling ---
    parser.add_argument("--profile", action="store_true", help="Run in profiling mode for a limited number of frames.")
    # ---------------------------------------------------------
    args = parser.parse_args()

    # Passer les arguments à main
    main(resume_training=args.resume, profile_mode=args.profile)