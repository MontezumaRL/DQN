import gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import time
import os

from config import *
from dqn import DQN
from replay_buffer import ReplayBuffer
from src.environment import MontezumaEnvironment
from utils import preprocess_frame

def evaluate_model(model_path):
    # Création de l'environnement
    env = MontezumaEnvironment(render_mode="human")

    # Détection du device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Création du réseau avec la même architecture
    policy_net = DQN((4, 84, 84), env.n_actions).to(device)

    # Chargement du modèle sauvegardé
    checkpoint = torch.load(model_path)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    policy_net.eval()  # Passage en mode évaluation

    # Boucle d'évaluation
    for episode in range(5):  # Par exemple, 5 épisodes d'évaluation
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Sélection de l'action de façon déterministe
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = policy_net(state_tensor).max(1)[1].item()

            # Exécution de l'action
            state, reward, done, info = env.step(action)
            episode_reward += reward

        print(f"Episode {episode}, Total Reward: {episode_reward}")

    env.close()

def train_montezuma():
    # Création de l'environnement
    env = MontezumaEnvironment(render_mode="human" if RENDER else None)
    
    # Détection du device (CPU ou GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialisation des réseaux
    policy_net = DQN((4, 84, 84), env.n_actions).to(device)
    target_net = DQN((4, 84, 84), env.n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Mode évaluation pour le réseau cible
    
    # Initialisation de l'optimiseur
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    
    # Initialisation du buffer de replay
    memory = ReplayBuffer(MEMORY_SIZE)
    
    # Initialisation des variables de suivi
    epsilon = EPSILON_START
    total_steps = 0
    episode_rewards = []
    
    # Boucle d'entraînement
    for episode in range(NUM_EPISODES):
        # Réinitialisation de l'environnement
        state = env.reset()
        episode_reward = 0
        episode_start_time = time.time()
        
        # Boucle d'un épisode
        while True:
            # Sélection de l'action (epsilon-greedy)
            if random.random() > epsilon:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net(state_tensor).max(1)[1].item()
            else:
                action = env.action_space.sample()
            
            # Exécution de l'action
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Stockage de l'expérience dans le buffer
            memory.push(state, action, reward, next_state, done)
            
            # Passage à l'état suivant
            state = next_state
            total_steps += 1
            
            # Apprentissage si le buffer contient assez d'expériences
            if len(memory) >= BATCH_SIZE:
                # Échantillonnage d'un batch d'expériences
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                
                # Conversion en tensors
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)
                
                # Calcul des Q-values actuelles
                current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Calcul des Q-values cibles
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    target_q_values = rewards + GAMMA * next_q_values * (1 - dones)
                
                # Calcul de la perte
                loss = nn.MSELoss()(current_q_values, target_q_values)
                
                # Optimisation
                optimizer.zero_grad()
                loss.backward()
                # Clip du gradient pour stabilité
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
            
            # Fin de l'épisode
            if done:
                break
        
        # Mise à jour du réseau cible
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Décroissance d'epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        # Calcul du temps d'épisode
        episode_duration = time.time() - episode_start_time
        
        # Enregistrement des récompenses
        episode_rewards.append(episode_reward)
        
        # Affichage des statistiques
        print(f"Episode {episode}, Reward: {episode_reward}, "
              f"Epsilon: {epsilon:.2f}, Duration: {episode_duration:.2f}s, "
              f"Steps: {total_steps}")
        
        # Sauvegarde périodique du modèle
        if episode % SAVE_INTERVAL == 0:
            save_dir = "models"
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'episode': episode,
                'policy_net_state_dict': policy_net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
                'total_steps': total_steps,
                'episode_rewards': episode_rewards,
            }, f"{save_dir}/montezuma_dqn_ep{episode}.pth")
            print(f"Model saved at episode {episode}")
    
    # Fermeture de l'environnement
    env.close()
    
    return policy_net
