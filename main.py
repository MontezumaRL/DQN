# main.py

import os
import torch
import numpy as np
import random
from src.training.train import train_montezuma

# Fixer les graines aléatoires pour la reproductibilité
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # Fixer la graine aléatoire
    set_seed(42)

    # Lancer l'entraînement
    train_montezuma()