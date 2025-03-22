# fichier dqn/src/training/train.py
from src.training.train import evaluate_model, train_montezuma

if __name__ == "__main__":
    checkpoint_path = "output/montezuma_dqn_ep10000_check2.pth"
    # x=105, y=148 En bas du niveau
    # (39, 148) Après la tete de mort
    # (21, 192) Devant la clé en haut de l'echelle

    new_start_x = 105  # nouvelle position x
    new_start_y = 148 # nouvelle position y

    #model = train_montezuma(
    #    checkpoint_path=checkpoint_path,
    #    start_x=new_start_x,
    #    start_y=new_start_y
    #)

    model = evaluate_model(
        model_path=checkpoint_path,
        start_x=new_start_x,
        start_y=new_start_y
    )

