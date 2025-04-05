# fichier: dqn/src/agent.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pickle # Pour sauvegarder/charger les normaliseurs
import os # Importer os pour vérifier l'existence des fichiers

from .networks import DQN_CNN, RNDNetwork
from .replay_buffer import ReplayBuffer
from .utils import RunningMeanStd, RNDObservationNormalizer
from . import config as cfg # Importer la configuration

class DQNAgent:
    def __init__(self, env, n_actions):
        self.env = env
        self.n_actions = n_actions
        self.device = cfg.DEVICE

        # --- Réseaux DQN ---
        self.policy_net = DQN_CNN(cfg.INPUT_SHAPE, n_actions).to(self.device)
        self.target_net = DQN_CNN(cfg.INPUT_SHAPE, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer_dqn = optim.Adam(self.policy_net.parameters(), lr=cfg.LEARNING_RATE_DQN, eps=1.5e-4)

        # --- Replay Buffer ---
        self.memory = ReplayBuffer(cfg.BUFFER_SIZE, self.device)

        # --- RND (si activé) ---
        self.use_rnd = cfg.USE_RND
        if self.use_rnd:
            print("--- Initializing RND ---") # DEBUG: Confirmer l'initialisation
            self.obs_normalizer = RNDObservationNormalizer(cfg.INPUT_SHAPE, clip=cfg.RND_OBS_CLIP)
            self.intrinsic_reward_rms = RunningMeanStd()
            self.rnd_target_net = RNDNetwork(cfg.INPUT_SHAPE, cfg.RND_OUTPUT_DIM).to(self.device)
            self.rnd_predictor_net = RNDNetwork(cfg.INPUT_SHAPE, cfg.RND_OUTPUT_DIM).to(self.device)
            print(f"RND Target Network on device: {next(self.rnd_target_net.parameters()).device}")
            print(f"RND Predictor Network on device: {next(self.rnd_predictor_net.parameters()).device}")
            for param in self.rnd_target_net.parameters():
                param.requires_grad = False
            self.rnd_target_net.eval()
            self.optimizer_rnd = optim.Adam(self.rnd_predictor_net.parameters(), lr=cfg.RND_LR, eps=1.5e-4)
            print(f"RND Target requires_grad=False: {all(not p.requires_grad for p in self.rnd_target_net.parameters())}")
            print(f"RND Predictor requires_grad=True: {all(p.requires_grad for p in self.rnd_predictor_net.parameters())}")
            print(f"RND Optimizer LR: {self.optimizer_rnd.param_groups[0]['lr']}")
            print("--- RND Initialized ---")

        # --- Exploration & State ---
        self.epsilon = cfg.EPSILON_START
        self.epsilon_decay = (cfg.EPSILON_START - cfg.EPSILON_FINAL) / cfg.EPSILON_DECAY_FRAMES
        self.total_steps = 0

    def choose_action(self, state_np):
        """Choisit une action en utilisant epsilon-greedy."""
        self.total_steps += 1
        self.epsilon = max(cfg.EPSILON_FINAL, cfg.EPSILON_START - self.epsilon_decay * self.total_steps)
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax(dim=1).item()
        return action

    def store_transition(self, state, action, extrinsic_reward, next_state, done):
        """Calcule la récompense RND et stocke la transition."""
        intrinsic_reward = 0.0
        if self.use_rnd:
            intrinsic_reward = self._compute_intrinsic_reward(next_state)
            # <<< PRINT ICI (Optionnel) : Pour voir la récompense finale ajoutée >>>
            # print(f"DEBUG RND STORE (Step {self.total_steps}): Final Intrinsic Reward = {intrinsic_reward:.8f}")

        total_reward = extrinsic_reward + cfg.INTRINSIC_REWARD_SCALE * intrinsic_reward
        self.memory.push(state, action, total_reward, next_state, done)

    def _compute_intrinsic_reward(self, next_state_np):
        """Calcule la récompense intrinsèque RND."""
        with torch.no_grad(): # Important pour ne pas calculer de gradients ici
            next_state_tensor = torch.from_numpy(next_state_np).float().unsqueeze(0) # Shape [1, C, H, W]

            # --- 1. Vérifier la normalisation de l'observation ---
            normalized_next_state_np = self.obs_normalizer(next_state_tensor.cpu().numpy()) # Appel __call__ met à jour les stats
            normalized_next_state = torch.from_numpy(normalized_next_state_np).float().to(self.device)
            
            # --- 2. Vérifier les embeddings ---
            target_embedding = self.rnd_target_net(normalized_next_state)
            # NOTE: Le prédicteur est aussi appelé en no_grad ici, car on calcule juste la *récompense*.
            # Les gradients pour le prédicteur ne sont nécessaires que lors de la mise à jour (update_networks).
            predictor_embedding = self.rnd_predictor_net(normalized_next_state)


            # --- 3. Vérifier l'erreur MSE brute (avant normalisation de récompense) ---
            # On utilise les embeddings calculés ci-dessus. .detach() est redondant dans no_grad mais sans danger.
            intrinsic_reward_raw = F.mse_loss(predictor_embedding, target_embedding.detach(), reduction='mean').item()


        # --- 4. Vérifier la normalisation de la récompense ---
        # L'update se fait HORS du with torch.no_grad() car rms utilise numpy
        self.intrinsic_reward_rms.update(np.array([intrinsic_reward_raw]))
        std_ri = self.intrinsic_reward_rms.std
        mean_ri = self.intrinsic_reward_rms.mean

        normalized_intrinsic_reward = intrinsic_reward_raw / (std_ri + 1e-8)


        clipped_intrinsic_reward = np.clip(normalized_intrinsic_reward, -cfg.RND_REWARD_CLIP, cfg.RND_REWARD_CLIP)


        # (Optionnel: stocker pour le log dans train.py si besoin)
        self.last_clipped_intrinsic_reward = clipped_intrinsic_reward

        return clipped_intrinsic_reward


    def update_networks(self):
        """Met à jour les réseaux DQN et RND."""
        if len(self.memory) < cfg.MIN_BUFFER_SIZE:
            return None, None

        states, actions, rewards, next_states, dones = self.memory.sample(cfg.BATCH_SIZE)

        # --- Mise à jour RND ---
        rnd_loss_value = None
        if self.use_rnd:
            # Normaliser les next_states du BATCH en utilisant les stats existantes (normalize, pas __call__)
            normalized_next_states_np = self.obs_normalizer.normalize(next_states.cpu().numpy())
            normalized_next_states = torch.from_numpy(normalized_next_states_np).float().to(self.device)

            # <<< PRINT (Optionnel) : Vérifier les stats des états normalisés du batch >>>
            # print(f"DEBUG RND UPDATE (Step {self.total_steps}): Batch Normalized States - Min={normalized_next_states.min():.4f}, Max={normalized_next_states.max():.4f}, Mean={normalized_next_states.mean():.4f}")

            # Calcul des embeddings pour la loss RND
            with torch.no_grad(): # Target ne nécessite pas de gradients
                target_embeddings = self.rnd_target_net(normalized_next_states)
            # Predictor nécessite des gradients pour la backpropagation
            predictor_embeddings = self.rnd_predictor_net(normalized_next_states)

            # Calcul de la loss RND (MSE moyenne sur le batch)
            rnd_loss = F.mse_loss(predictor_embeddings, target_embeddings.detach(), reduction='mean')

            # Mise à jour du réseau prédicteur RND
            self.optimizer_rnd.zero_grad()
            rnd_loss.backward()
            # <<< PRINT (Optionnel) : Vérifier les gradients avant clipping/step >>>
            # grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(self.rnd_predictor_net.parameters(), max_norm=float('inf')) # Calculer sans clipper
            # print(f"DEBUG RND UPDATE (Step {self.total_steps}): Predictor Grad Norm (Before Clip) = {grad_norm_before_clip:.6f}")

            torch.nn.utils.clip_grad_norm_(self.rnd_predictor_net.parameters(), max_norm=0.5)
            self.optimizer_rnd.step()
            rnd_loss_value = rnd_loss.item()

        # --- Mise à jour DQN (DDQN) ---
        # (Le reste de la fonction DQN est inchangé)
        with torch.no_grad():
            best_actions_next = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            q_values_next_target = self.target_net(next_states).gather(1, best_actions_next)
            q_target = rewards + cfg.GAMMA * q_values_next_target * (1 - dones) # Utilise la récompense totale du buffer

        q_values = self.policy_net(states).gather(1, actions)
        dqn_loss = F.smooth_l1_loss(q_values, q_target)

        self.optimizer_dqn.zero_grad()
        dqn_loss.backward()
        # <<< PRINT (Optionnel) : Vérifier si les gradients DQN affectent RND par erreur >>>
        # rnd_pred_grad_sum_after_dqn = sum(p.grad.abs().sum().item() for p in self.rnd_predictor_net.parameters() if p.grad is not None)
        # print(f"DEBUG RND UPDATE (Step {self.total_steps}): RND Predictor Grad Sum After DQN backward = {rnd_pred_grad_sum_after_dqn}") # Devrait être 0 si zero_grad a bien marché

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer_dqn.step()

        # --- Mettre à jour le Target Network (DQN) ---
        if self.total_steps % cfg.TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return dqn_loss.item(), rnd_loss_value



    # ===================================================================
    # MODIFICATIONS DANS save_models et load_models CI-DESSOUS
    # ===================================================================

    def save_models(self):
        """Sauvegarde les poids des modèles, l'état des optimiseurs et l'état de l'entraînement."""
        print(f"Saving models and training state at step {self.total_steps}...")
        try:
            # Sauvegarde des réseaux
            torch.save(self.policy_net.state_dict(), cfg.MODEL_SAVE_PATH)
            if self.use_rnd:
                torch.save(self.rnd_predictor_net.state_dict(), cfg.RND_SAVE_PATH)

            # Préparation de l'état de l'entraînement
            training_state = {
                'optimizer_dqn': self.optimizer_dqn.state_dict(),
                'total_steps': self.total_steps,   # <-- Sauvegarde total_steps
                'epsilon': self.epsilon            # <-- Sauvegarde epsilon
            }
            if self.use_rnd:
                training_state['optimizer_rnd'] = self.optimizer_rnd.state_dict()

            # Sauvegarde de l'état de l'entraînement (inclut optimizers, steps, epsilon)
            torch.save(training_state, cfg.OPTIM_SAVE_PATH)

            # Sauvegarde des normaliseurs RND (séparément car pickle)
            if self.use_rnd:
                with open(cfg.NORM_SAVE_PATH, 'wb') as f:
                    pickle.dump({
                        'obs_normalizer': self.obs_normalizer,
                        'intrinsic_reward_rms': self.intrinsic_reward_rms
                    }, f)

            print("Models, optimizers, normalizers, and training state saved successfully.")

        except Exception as e:
            print(f"Error during saving: {e}")


    def load_models(self):
        """Charge les poids des modèles, l'état des optimiseurs et l'état de l'entraînement."""
        models_loaded = False
        try:
            # --- Chargement des réseaux ---
            if os.path.exists(cfg.MODEL_SAVE_PATH):
                self.policy_net.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=self.device))
                self.target_net.load_state_dict(self.policy_net.state_dict()) # MAJ target net
                print(f"Loaded DQN models from {cfg.MODEL_SAVE_PATH}")
                models_loaded = True
                if self.use_rnd:
                    if os.path.exists(cfg.RND_SAVE_PATH):
                         self.rnd_predictor_net.load_state_dict(torch.load(cfg.RND_SAVE_PATH, map_location=self.device))
                         print(f"Loaded RND predictor model from {cfg.RND_SAVE_PATH}")
                    else:
                         print(f"Warning: RND predictor model file not found: {cfg.RND_SAVE_PATH}")
            else:
                print(f"DQN model file not found: {cfg.MODEL_SAVE_PATH}. Starting from scratch or expecting only training state.")

            # --- Chargement de l'état de l'entraînement ---
            if os.path.exists(cfg.OPTIM_SAVE_PATH):
                training_state = torch.load(cfg.OPTIM_SAVE_PATH, map_location=self.device)
                print(f"Loading training state from {cfg.OPTIM_SAVE_PATH}...")

                # Charger les optimiseurs
                if 'optimizer_dqn' in training_state:
                    self.optimizer_dqn.load_state_dict(training_state['optimizer_dqn'])
                    print("  Loaded DQN optimizer state.")
                else:
                    print("  Warning: DQN optimizer state not found in training state file.")

                if self.use_rnd and 'optimizer_rnd' in training_state:
                     self.optimizer_rnd.load_state_dict(training_state['optimizer_rnd'])
                     print("  Loaded RND optimizer state.")
                elif self.use_rnd:
                     print("  Warning: RND optimizer state not found in training state file.")

                # Charger total_steps et epsilon
                if 'total_steps' in training_state:
                    self.total_steps = training_state['total_steps']
                    print(f"  Resuming from total_steps: {self.total_steps}")
                    # Recalculer epsilon basé sur les steps chargés, au cas où epsilon n'est pas sauvegardé
                    # ou pour assurer la cohérence si on modifie la formule de decay
                    self.epsilon = max(cfg.EPSILON_FINAL, cfg.EPSILON_START - self.epsilon_decay * self.total_steps)
                    print(f"  Recalculated epsilon based on loaded steps: {self.epsilon:.4f}")
                    #self.epsilon = 0.3
                    #print(f"  Forcing epsilon: {self.epsilon:.4f}") 
                    # Si epsilon est explicitement sauvegardé, on peut l'utiliser, mais recalculer est plus sûr
                    if 'epsilon' in training_state:
                         # Optionnel: utiliser l'epsilon sauvegardé
                         # self.epsilon = training_state['epsilon']
                         # print(f"  Using saved epsilon: {self.epsilon:.4f}")
                         pass # On préfère le recalculer par défaut

                else:
                     print("  Warning: 'total_steps' not found in training state file. Steps/Epsilon might be incorrect.")

                models_loaded = True # Considère comme chargé si l'état de l'entraînement existe
            else:
                 print(f"Training state file not found: {cfg.OPTIM_SAVE_PATH}")


            # --- Chargement des normaliseurs RND ---
            if self.use_rnd:
                if os.path.exists(cfg.NORM_SAVE_PATH):
                    try:
                        with open(cfg.NORM_SAVE_PATH, 'rb') as f:
                            norm_data = pickle.load(f)
                            self.obs_normalizer = norm_data['obs_normalizer']
                            self.intrinsic_reward_rms = norm_data['intrinsic_reward_rms']
                        print(f"Loaded RND normalizers from {cfg.NORM_SAVE_PATH}")
                        models_loaded = True # Renforce le fait que des données ont été chargées
                    except Exception as e:
                        print(f"  Warning: Could not load normalizers from {cfg.NORM_SAVE_PATH}: {e}")
                else:
                    print(f"RND normalizer file not found: {cfg.NORM_SAVE_PATH}")


            # --- Finalisation ---
            if models_loaded:
                print("Finished loading saved data.")
                # Mettre les réseaux dans le bon mode
                self.policy_net.train()
                self.target_net.eval()
                if self.use_rnd:
                    self.rnd_predictor_net.train()
                    self.rnd_target_net.eval() # Assurer que le target RND est bien en eval
            else:
                 print("No saved data found or loaded. Starting training from scratch.")


        except FileNotFoundError: # Redondant avec os.path.exists mais sécurité
            print("Starting from scratch as some essential files were not found.")
        except Exception as e:
            print(f"An error occurred during loading: {e}. Starting from scratch.")