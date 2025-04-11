# fichier: dqn/src/utils.py
import numpy as np
import cv2
# import torch # torch n'est pas utilisé directement ici, peut être retiré si non nécessaire ailleurs

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Prétraite une frame du jeu Atari pour l'entrée du réseau neuronal.
    - Convertit en niveaux de gris.
    - Redimensionne à 42x42 pixels.
    - Normalise les valeurs des pixels dans l'intervalle [0.0, 1.0].

    Args:
        frame (np.ndarray): La frame brute de l'environnement (H, W, C) ou (H, W).

    Returns:
        np.ndarray: La frame prétraitée (42, 42) de type float32.
    """
    if frame is None:
        # Gérer le cas où la frame est None (peut arriver à la fin d'un épisode)
        print("Warning: preprocess_frame received None, returning zeros.")
        return np.zeros((42, 42), dtype=np.float32)

    # S'assurer que la frame est en niveaux de gris
    if frame.ndim == 3 and frame.shape[2] == 3: # Vérifie si c'est une image couleur RGB
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    elif frame.ndim == 2: # Déjà en niveaux de gris
        frame_gray = frame
    else: # Format inattendu
         print(f"Warning: Unexpected frame shape {frame.shape} in preprocess_frame, returning zeros.")
         return np.zeros((42, 42), dtype=np.float32)

    # Redimensionner en 42x42
    frame_resized = cv2.resize(frame_gray, (42, 42), interpolation=cv2.INTER_AREA)

    # Normaliser les pixels entre 0 et 1 et convertir en float32
    frame_normalized = np.array(frame_resized, dtype=np.float32) / 255.0

    return frame_normalized

# Classe pour calculer la moyenne et l'écart-type courants (running mean/std)
class RunningMeanStd:
    """
    Calcule la moyenne et l'écart-type d'un flux de données en utilisant
    l'algorithme parallèle de Welford. Maintient la précision avec float64.

    Args:
        epsilon (float): Petite valeur ajoutée au compteur pour la stabilité numérique initiale.
        shape (tuple): Forme des données attendues (ex: () pour des scalaires, (C, H, W) pour des images).
    """
    def __init__(self, epsilon=1e-4, shape=()):
        # Utilise float64 pour la précision des calculs internes
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64) # Initialiser var à 1, pas 0
        self.count = epsilon

    def update(self, x: np.ndarray):
        """Met à jour la moyenne et la variance avec un nouveau batch de données."""
        # x doit être un np.ndarray
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        # Algorithme de Welford (version parallèle)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        # M2 est la somme des carrés des différences par rapport à la nouvelle moyenne
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        # Mettre à jour les attributs
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self) -> np.ndarray:
        """Retourne l'écart-type courant."""
        # Ajoute un petit epsilon pour éviter sqrt(0) ou valeurs négatives très petites dues à l'imprécision
        return np.sqrt(np.maximum(self.var, 1e-8))

    @property
    def mean_val(self) -> np.ndarray:
         """Retourne la moyenne courante."""
         return self.mean


# Classe pour normaliser les observations pour RND en utilisant RunningMeanStd
class RNDObservationNormalizer:
    """
    Normalise les observations pour les réseaux RND en utilisant RunningMeanStd.
    Permet de normaliser avec ou sans mise à jour des statistiques.
    Applique également un clipping.

    Args:
        input_shape (tuple): Forme des observations à normaliser.
        clip (float): Valeur absolue maximale pour les observations normalisées.
    """
    def __init__(self, input_shape, clip=5.0):
        self.rms = RunningMeanStd(shape=input_shape)
        self.clip = clip
        self._epsilon = 1e-8 # Epsilon pour la division par std

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Met à jour les statistiques, puis normalise et clippe l'observation."""
        # D'abord, mettre à jour les statistiques avec la nouvelle observation (ou batch)
        self.rms.update(obs)
        # Ensuite, normaliser et clipper
        return self._normalize_clip(obs)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalise et clippe l'observation en utilisant les statistiques actuelles, sans les mettre à jour."""
        return self._normalize_clip(obs)

    def _normalize_clip(self, obs: np.ndarray) -> np.ndarray:
        """Fonction helper pour normaliser et clipper."""
        # Normalisation: (observation - moyenne_courante) / (ecart_type_courant + epsilon)
        normalized_obs = (obs - self.rms.mean_val) / (self.rms.std + self._epsilon)
        # Clipping pour éviter les valeurs extrêmes
        clipped_obs = np.clip(normalized_obs, -self.clip, self.clip)
        return clipped_obs

    # Optionnel: Méthodes pour obtenir directement mean/std si nécessaire pour du logging externe
    @property
    def mean(self):
        return self.rms.mean_val

    @property
    def std(self):
        return self.rms.std