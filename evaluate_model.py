import argparse
from src.evaluation.evaluate import evaluate_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation d'un modèle DQN")
    parser.add_argument("-m", type=str, required=True, help="Chemin vers le fichier du modèle", dest="model")
    parser.add_argument("-e", type=int, default=5, help="Nombre d'épisodes d'évaluation", dest="episodes")

    args = parser.parse_args()

    print(f"Évaluation du modèle: {args.model}")
    evaluate_model(args.model, args.episodes)