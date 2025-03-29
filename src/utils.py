# fichier: dqn/src/utils.py
import numpy as np
import cv2
import torch

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Prétraite une frame du jeu Atari.
    - Convertit en niveaux de gris
    - Redimensionne à 84x84
    - Normalise les pixels entre 0 et 1
    """
    if frame is None:
        # Gérer le cas où la frame est None (peut arriver à la fin)
        return np.zeros((84, 84), dtype=np.float32)
    if frame.ndim == 3 and frame.shape[2] == 3: # Vérifie si c'est une image couleur
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    elif frame.ndim == 2: # Déjà en niveaux de gris
        frame_gray = frame
    else: # Format inattendu, retourne un tableau noir
         print(f"Warning: Unexpected frame shape {frame.shape}, returning zeros.")
         return np.zeros((84, 84), dtype=np.float32)

    frame_resized = cv2.resize(frame_gray, (84, 84), interpolation=cv2.INTER_AREA)
    frame_normalized = np.array(frame_resized, dtype=np.float32) / 255.0
    return frame_normalized

# Classe pour la normalisation (basée sur Welford's algorithm)
class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var)

# Classe pour normaliser les observations pour RND
class RNDObservationNormalizer:
    def __init__(self, input_shape, clip=5.0):
        self.rms = RunningMeanStd(shape=input_shape)
        self.clip = clip

    def __call__(self, obs):
        self.rms.update(obs)
        # Normalisation: (obs - mean) / std
        # On clip pour éviter les valeurs extrêmes
        normalized_obs = np.clip((obs - self.rms.mean) / (self.rms.std + 1e-8),
                                 -self.clip, self.clip)
        return normalized_obs

    def normalize(self, obs):
        # Normalise sans mettre à jour les stats (utile à l'inférence)
        return np.clip((obs - self.rms.mean) / (self.rms.std + 1e-8),
                       -self.clip, self.clip)