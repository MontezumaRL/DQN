# DDQN

Pour exécuter le code, il faut d'abord installer les dépendances. Pour cela, exécutez la commande suivante :
(Conseil créer un environnement virtuel python avant d'installer les dépendances)

```bash
pip install -r requirements.txt
```
Ensuite, vous pouvez exécuter le code en utilisant les commandes suivantes :

Pour l'entraînement :
```bash
python train.py
```
Pour continuer l'entraînement :
```bash
python train.py --resume
```

Pour l'évaluation : 
```bash
python evaluate.py
```
Nécessite un modèle entraîné, les fichiers dqn_montezuma_model.pth, normalizers_montezuma.pkl et optimizers_montezuma.pth et rnd_montezuma_model.pth doivent être présents dans le répertoire d'exécution.

Pour modifier les hyperparamètres, vous pouvez le faire directement dans le fichier config.py.
Pour modifier les paramètres d'entraînement, vous pouvez le faire directement dans le fichier train.py.
Pour modifier les paramètres d'évaluation, vous pouvez le faire directement dans le fichier evaluate.py.