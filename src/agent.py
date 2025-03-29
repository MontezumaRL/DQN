# fichier: dqn/src/agent.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pickle # Pour sauvegarder/charger les normaliseurs

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
        # Copier les poids initiaux dans le target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Mettre le target net en mode évaluation

        self.optimizer_dqn = optim.Adam(self.policy_net.parameters(), lr=cfg.LEARNING_RATE_DQN, eps=1.5e-4) # Ajout epsilon pour Adam (souvent recommandé pour RL)

        # --- Replay Buffer ---
        self.memory = ReplayBuffer(cfg.BUFFER_SIZE, self.device)

        # --- RND (si activé) ---
        self.use_rnd = cfg.USE_RND
        if self.use_rnd:
            # Normaliseur pour les états entrant dans RND
            self.obs_normalizer = RNDObservationNormalizer(cfg.INPUT_SHAPE, clip=cfg.RND_OBS_CLIP)
            # Normaliseur pour les récompenses intrinsèques brutes
            self.intrinsic_reward_rms = RunningMeanStd()

            # Réseaux RND
            self.rnd_target_net = RNDNetwork(cfg.INPUT_SHAPE, cfg.RND_OUTPUT_DIM).to(self.device)
            self.rnd_predictor_net = RNDNetwork(cfg.INPUT_SHAPE, cfg.RND_OUTPUT_DIM).to(self.device)

            # Le réseau target RND est fixe (pas d'entraînement)
            for param in self.rnd_target_net.parameters():
                param.requires_grad = False
            self.rnd_target_net.eval()

            # Optimizer pour le prédicteur RND
            self.optimizer_rnd = optim.Adam(self.rnd_predictor_net.parameters(), lr=cfg.RND_LR, eps=1.5e-4)

        # --- Exploration ---
        self.epsilon = cfg.EPSILON_START
        self.epsilon_decay = (cfg.EPSILON_START - cfg.EPSILON_FINAL) / cfg.EPSILON_DECAY_FRAMES
        self.total_steps = 0

    def choose_action(self, state_np):
        """Choisit une action en utilisant epsilon-greedy."""
        self.total_steps += 1

        # Mettre à jour epsilon
        self.epsilon = max(cfg.EPSILON_FINAL, cfg.EPSILON_START - self.epsilon_decay * self.total_steps)

        if random.random() < self.epsilon:
            # Action aléatoire
            action = self.env.action_space.sample()
        else:
            # Action basée sur la politique (Q-values)
            with torch.no_grad():
                state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax(dim=1).item() # Prend l'action avec la Q-value max

        return action

    def store_transition(self, state, action, extrinsic_reward, next_state, done):
        """Calcule la récompense RND et stocke la transition."""
        intrinsic_reward = 0.0
        if self.use_rnd:
            intrinsic_reward = self._compute_intrinsic_reward(next_state)

        # Combinaison des récompenses
        # On ne normalise PAS la récompense extrinsèque ici, seulement l'intrinsèque
        total_reward = extrinsic_reward + cfg.INTRINSIC_REWARD_SCALE * intrinsic_reward

        self.memory.push(state, action, total_reward, next_state, done)

    def _compute_intrinsic_reward(self, next_state_np):
        """Calcule la récompense intrinsèque RND."""
        with torch.no_grad():
            # 1. Normaliser l'état suivant pour RND
            #    Convertir en tensor, ajouter dim batch, normaliser, reconvertir en numpy si besoin
            next_state_tensor = torch.from_numpy(next_state_np).float().unsqueeze(0) # Shape [1, C, H, W]
            normalized_next_state_np = self.obs_normalizer(next_state_tensor.cpu().numpy()) # Normaliseur attend numpy
            normalized_next_state = torch.from_numpy(normalized_next_state_np).float().to(self.device) # Retour sur device

            # 2. Obtenir les embeddings des réseaux RND
            target_embedding = self.rnd_target_net(normalized_next_state)
            predictor_embedding = self.rnd_predictor_net(normalized_next_state)

            # 3. Calculer l'erreur MSE (récompense brute)
            #    On prend la moyenne sur la dimension de l'embedding
            intrinsic_reward_raw = F.mse_loss(predictor_embedding, target_embedding, reduction='mean').item() # Moyenne sur le batch (ici 1) et l'embedding

        # 4. Mettre à jour les stats de normalisation de la récompense intrinsèque
        self.intrinsic_reward_rms.update(np.array([intrinsic_reward_raw]))

        # 5. Normaliser la récompense intrinsèque
        #    r_i = r_i_raw / (std_r_i + eps)
        #    On ajoute un petit epsilon pour la stabilité numérique
        std_ri = self.intrinsic_reward_rms.std
        normalized_intrinsic_reward = intrinsic_reward_raw / (std_ri + 1e-8)

        # 6. Clipper la récompense normalisée (optionnel mais souvent utile)
        clipped_intrinsic_reward = np.clip(normalized_intrinsic_reward, -cfg.RND_REWARD_CLIP, cfg.RND_REWARD_CLIP)

        return clipped_intrinsic_reward


    def update_networks(self):
        """Met à jour les réseaux DQN et RND."""
        if len(self.memory) < cfg.MIN_BUFFER_SIZE:
            return None, None # Pas assez de données pour entraîner

        # --- Échantillonner un batch ---
        states, actions, rewards, next_states, dones = self.memory.sample(cfg.BATCH_SIZE)

        # --- Mise à jour RND (si activé) ---
        rnd_loss = None
        if self.use_rnd:
            # Normaliser les next_states pour RND
            normalized_next_states_np = self.obs_normalizer.normalize(next_states.cpu().numpy()) # Utilise normalize (pas d'update des stats)
            normalized_next_states = torch.from_numpy(normalized_next_states_np).float().to(self.device)

            # Calculer les embeddings
            with torch.no_grad():
                target_embeddings = self.rnd_target_net(normalized_next_states)
            predictor_embeddings = self.rnd_predictor_net(normalized_next_states)

            # Calculer la loss RND (MSE)
            # On veut que la loss soit proportionnelle à la somme des erreurs sur le batch
            rnd_loss = F.mse_loss(predictor_embeddings, target_embeddings.detach(), reduction='mean') # Moyenne sur le batch

            # Mettre à jour le prédicteur RND
            self.optimizer_rnd.zero_grad()
            rnd_loss.backward()
            # Gradient clipping (peut aider)
            torch.nn.utils.clip_grad_norm_(self.rnd_predictor_net.parameters(), max_norm=0.5)
            self.optimizer_rnd.step()

            rnd_loss = rnd_loss.item() # Pour logging

        # --- Mise à jour DQN (DDQN) ---

        # 1. Calculer Q(s, a) pour les états et actions du batch
        #    policy_net(states) donne les Q-values pour toutes les actions.
        #    .gather(1, actions) sélectionne la Q-value pour l'action spécifique prise.
        q_values = self.policy_net(states).gather(1, actions)

        # 2. Calculer la target Q-value (Double DQN)
        with torch.no_grad():
            # a. Sélectionner la meilleure action a' pour s' selon le policy_net
            #    argmax trouve l'index de l'action max. keepdim=True garde la dim [batch, 1]
            best_actions_next = self.policy_net(next_states).argmax(dim=1, keepdim=True)

            # b. Évaluer cette action a' avec le target_net: Q_target(s', argmax_a' Q_policy(s', a'))
            q_values_next_target = self.target_net(next_states).gather(1, best_actions_next)

            # c. Calculer la target: r + gamma * Q_target(...) * (1 - done)
            #    (1 - dones) met la Q-value future à 0 si l'épisode est terminé
            q_target = rewards + cfg.GAMMA * q_values_next_target * (1 - dones)

        # 3. Calculer la loss DQN (Smooth L1 Loss ou MSE Loss)
        #    Smooth L1 est souvent plus robuste aux outliers que MSE
        dqn_loss = F.smooth_l1_loss(q_values, q_target)
        # Ou MSE: dqn_loss = F.mse_loss(q_values, q_target)

        # 4. Mettre à jour le policy_net
        self.optimizer_dqn.zero_grad()
        dqn_loss.backward()
        # Gradient clipping (important pour la stabilité)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0) # max_norm=1.0 est courant
        self.optimizer_dqn.step()

        # --- Mettre à jour le Target Network (périodiquement) ---
        if self.total_steps % cfg.TARGET_UPDATE_FREQ == 0:
            #print(f"Updating target network at step {self.total_steps}")
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return dqn_loss.item(), rnd_loss # Retourne les pertes pour logging

    def save_models(self):
        """Sauvegarde les poids des modèles et l'état des optimiseurs."""
        print(f"Saving models at step {self.total_steps}...")
        torch.save(self.policy_net.state_dict(), cfg.MODEL_SAVE_PATH)
        optimizers_state = {
            'optimizer_dqn': self.optimizer_dqn.state_dict(),
        }
        if self.use_rnd:
            torch.save(self.rnd_predictor_net.state_dict(), cfg.RND_SAVE_PATH)
            optimizers_state['optimizer_rnd'] = self.optimizer_rnd.state_dict()
            # Sauvegarder les normaliseurs RND (avec pickle car ils contiennent des états numpy)
            with open(cfg.NORM_SAVE_PATH, 'wb') as f:
                pickle.dump({
                    'obs_normalizer': self.obs_normalizer,
                    'intrinsic_reward_rms': self.intrinsic_reward_rms
                }, f)

        torch.save(optimizers_state, cfg.OPTIM_SAVE_PATH)
        print("Models, optimizers, and normalizers saved.")


    def load_models(self):
        """Charge les poids des modèles et l'état des optimiseurs."""
        try:
            print("Loading models...")
            self.policy_net.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict()) # Important: MAJ target net aussi

            optimizers_state = torch.load(cfg.OPTIM_SAVE_PATH, map_location=self.device)
            self.optimizer_dqn.load_state_dict(optimizers_state['optimizer_dqn'])

            if self.use_rnd:
                self.rnd_predictor_net.load_state_dict(torch.load(cfg.RND_SAVE_PATH, map_location=self.device))
                if 'optimizer_rnd' in optimizers_state:
                     self.optimizer_rnd.load_state_dict(optimizers_state['optimizer_rnd'])
                # Charger les normaliseurs RND
                with open(cfg.NORM_SAVE_PATH, 'rb') as f:
                    norm_data = pickle.load(f)
                    self.obs_normalizer = norm_data['obs_normalizer']
                    self.intrinsic_reward_rms = norm_data['intrinsic_reward_rms']

            self.policy_net.train() # Assure que le policy net est en mode train
            self.target_net.eval() # Assure que le target net est en mode eval
            if self.use_rnd:
                self.rnd_predictor_net.train() # Assure que le predictor est en mode train

            print("Models, optimizers, and normalizers loaded successfully.")
            # Note: On ne charge pas total_steps ou epsilon ici, on reprend une nouvelle phase d'explo/decay si besoin
            # Si tu veux reprendre exactement où tu étais, il faut sauvegarder/charger aussi self.total_steps et self.epsilon
        except FileNotFoundError:
            print("No saved models found, starting from scratch.")
        except Exception as e:
            print(f"Error loading models: {e}. Starting from scratch.")