# DDQN

Pour exécuter le code, il faut d'abord installer les dépendances. Pour cela, exécutez la commande suivante :
(Conseil créer un environnement virtuel python avant d'installer les dépendances)

```bash
make install
```
Cela va automatiquement créer un environnement virtuel python et installer toutes les dépendances nécessaires.

Ensuite, vous pouvez exécuter le code en utilisant les commandes suivantes :

Pour commencer un entraînement :
```bash
make train
```
Pour continuer un entraînement déjà commencé :
```bash
make resume
```
Pour l'évaluation d'un modèle (exemple fourni): 
```bash
make evaluate
```
Nécessite un modèle entraîné (fourni dans le zip), les fichiers dqn_montezuma_model.pth, normalizers_montezuma.pkl et optimizers_montezuma.pth et rnd_montezuma_model.pth doivent être présents dans le répertoire d'exécution.

Pour modifier les hyperparamètres, vous pouvez le faire directement dans le fichier config.py.
Pour modifier les paramètres d'entraînement, vous pouvez le faire directement dans le fichier train.py.
Pour modifier les paramètres d'évaluation, vous pouvez le faire directement dans le fichier evaluate.py.