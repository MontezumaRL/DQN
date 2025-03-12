import cv2

def preprocess_frame(frame):
    # Convertir en gris et redimensionner
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))
    return frame / 255.0