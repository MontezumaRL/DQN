# fichier dqn/src/utils.py
import numpy as np

from PIL import Image

def preprocess_frame(frame):
    # Convertir en PIL Image
    img = Image.fromarray(frame)
    # Convertir en niveaux de gris
    img = img.convert('L')
    # Redimensionner
    img = img.resize((42, 42), Image.Resampling.BILINEAR)
    # Convertir en numpy array
    frame = np.array(img)

    # Normaliser et convertir en 3-bit
    frame = frame / 255.0
    frame = frame * 7
    frame = np.round(frame)
    frame = frame / 7

    return frame