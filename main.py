# fichier dqn/src/training/train.py
from src.training.train_dqn import train_montezuma
from src.training.train_ddqn import train_montezuma_dueling

if __name__ == "__main__":
   checkpoint_path = "output/ddqn/checkpoint_ep4500.pth"
   # x=105, y=148 En bas du niveau
   # (39, 148) Après la tete de mort
   # (21, 192) Devant la clé en haut de l'echelle

   new_start_x = 39  # nouvelle position x
   new_start_y = 148 # nouvelle position y

   #model = train_montezuma(
   #   checkpoint_path=None,
   #   start_x=new_start_x,
   #   start_y=new_start_y
   #)

   model = train_montezuma_dueling(
      checkpoint_path=None,
      start_x=new_start_x,
      start_y=new_start_y
   )

