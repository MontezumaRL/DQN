import os
import pytest
import numpy as np
import matplotlib.pyplot as plt

from src.environment import MontezumaEnvironment
from src.utils import preprocess_frame

@pytest.fixture
def env():
    """Fixture pour créer et fermer l'environnement automatiquement"""
    environment = MontezumaEnvironment(render_mode="rgb_array")
    yield environment
    environment.close()

@pytest.fixture(scope="session", autouse=True)
def setup_test_output_dir():
    test_output_dir = "test_outputs"
    os.makedirs(test_output_dir, exist_ok=True)
    return test_output_dir

def test_preprocess_frame(env, setup_test_output_dir):
    """Teste la fonction de prétraitement des frames"""
    # Obtenir une frame brute
    env.reset()
    raw_frame = env.env.render()

    # Prétraiter la frame
    processed_frame = preprocess_frame(raw_frame)

    # Vérifier la forme
    assert processed_frame.shape == (84, 84)

    # Vérifier que les valeurs sont normalisées entre 0 et 1
    assert np.all(processed_frame >= 0) and np.all(processed_frame <= 1)

    # Sauvegarder les images pour inspection visuelle
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Raw Frame")
    plt.imshow(raw_frame)

    plt.subplot(1, 2, 2)
    plt.title("Processed Frame")
    plt.imshow(processed_frame, cmap='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(setup_test_output_dir, "preprocess_test.png"))
    plt.close()

def test_frame_stack_consistency(env, setup_test_output_dir):
    """Teste la cohérence du stack de frames"""
    env.reset()

    # Exécuter plusieurs actions
    for _ in range(3):
        env.step(1)  # Action RIGHT

    # Vérifier que le stack contient 4 frames
    assert len(env.frame_stack) == 4

    # Vérifier que les frames sont différentes
    frames = list(env.frame_stack)
    for i in range(len(frames) - 1):
        # Calculer la différence moyenne entre frames consécutives
        diff = np.mean(np.abs(frames[i] - frames[i+1]))
        # La différence ne devrait pas être nulle (frames identiques)
        assert diff > 0

    # Sauvegarder le stack pour inspection visuelle
    env.display_frame_stack(save_path=os.path.join(setup_test_output_dir, "frame_stack_consistency.png"))

@pytest.mark.visual
def test_visual_preprocess():
    """Test visuel du prétraitement"""
    env = MontezumaEnvironment(render_mode="rgb_array")
    try:
        env.reset()

        # Exécuter quelques actions
        for _ in range(5):
            env.step(np.random.randint(0, env.n_actions))

        # Afficher le stack de frames
        env.display_frame_stack(save_path="test_outputs/preprocess_visual_test.png")
        print(f"Stack de frames sauvegardé dans test_outputs/preprocess_visual_test.png")
    finally:
        env.close()