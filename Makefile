# Makefile pour gérer l'environnement virtuel Python et les dépendances

# --- Configuration ---
VENV_DIR := .venv
PYTHON ?= python3
REQUIREMENTS := requirements.txt

# Détermine les chemins dans l'environnement virtuel (adapte pour Windows si nécessaire)
ifeq ($(OS),Windows_NT)
	PYTHON_VENV := $(VENV_DIR)/Scripts/python.exe
	ACTIVATE_CMD := $(VENV_DIR)/Scripts/activate
	# Note: La commande exacte sous Windows dépend du shell (cmd, PowerShell, Git Bash)
	# Pour Git Bash, utiliser: source $(VENV_DIR)/Scripts/activate
else
	PYTHON_VENV := $(VENV_DIR)/bin/python
	ACTIVATE_CMD := source $(VENV_DIR)/bin/activate
endif

INSTALL_STAMP := $(VENV_DIR)/install.stamp

# --- Cibles Principales ---

# Cible par défaut : crée l'environnement et installe les dépendances
.PHONY: all
all: $(INSTALL_STAMP)

# Cible pour créer l'environnement virtuel
$(PYTHON_VENV):
	@echo ">>> Création de l'environnement virtuel dans $(VENV_DIR)..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo ">>> Environnement virtuel créé."
	@# Mettre à jour pip
	$(PYTHON_VENV) -m pip install --upgrade pip
	@echo ">>> Pip mis à jour dans l'environnement virtuel."

# Cible pour installer/mettre à jour les dépendances
$(INSTALL_STAMP): $(PYTHON_VENV) $(REQUIREMENTS)
	@echo ">>> Installation/Mise à jour des dépendances depuis $(REQUIREMENTS)..."
	$(PYTHON_VENV) -m pip install -r $(REQUIREMENTS)
	@echo ">>> Dépendances installées."
	@touch $@

# --- Cibles Utilitaires ---

# Cible explicite pour installer (alias de 'all')
.PHONY: install
install: $(INSTALL_STAMP)

# Cible pour créer l'environnement virtuel uniquement (si nécessaire)
.PHONY: venv
venv: $(PYTHON_VENV)

# Cible pour nettoyer (supprimer l'environnement virtuel)
.PHONY: clean
clean:
	@echo ">>> Suppression de l'environnement virtuel $(VENV_DIR)..."
	@$(PYTHON) -c "import shutil; shutil.rmtree('$(VENV_DIR)', ignore_errors=True)"
	@echo ">>> Nettoyage terminé."

.PHONY: train
train: $(INSTALL_STAMP) # Assure que le venv et les deps sont prêts
	@echo ">>> Exécution de 'python train.py' dans l'environnement virtuel..."
	$(PYTHON_VENV) train.py

.PHONY: resume
resume: $(INSTALL_STAMP) # Assure que le venv et les deps sont prêts
	@echo ">>> Exécution de 'python train.py --resume' dans l'environnement virtuel"
	$(PYTHON_VENV) train.py --resume

.PHONY: evaluate
evaluate: $(INSTALL_STAMP) # Assure que le venv et les deps sont prêts
	@echo ">>> Exécution de 'python evaluate.py' dans l'environnement virtuel..."
	$(PYTHON_VENV) evaluate.py $(ARGS)


# Cible pour afficher l'aide
.PHONY: help
help:
	@echo "Makefile pour projet Python"
	@echo ""
	@echo "Cibles principales:"
	@echo "  make          : (ou make all) Crée env virtuel + installe dépendances."
	@echo "  make install  : Installe/met à jour les dépendances (crée l'env si besoin)."
	@echo ""
	@echo "Gestion de l'environnement:"
	@echo "  make venv     : Crée l'environnement virtuel $(VENV_DIR) (si besoin)."
	@echo "  make clean    : Supprime l'environnement virtuel ($(VENV_DIR))."
	@echo ""
	@echo "Activation & Exécution:"
	@echo "  make train         : Exécute 'python train.py' DANS l'env virtuel."
	@echo "  make train ARGS="--resume"        : Exécute 'python train.py' DANS l'env virtuel et reprend l'entraînement."
	@echo "  make evaluate         : Exécute 'python evaluate.py' DANS l'env virtuel."
	@echo ""
	@echo "Configuration (surcharge possible):"
	@echo "  PYTHON       : ($(PYTHON)) Interpréteur Python à utiliser."
	@echo "  VENV_DIR     : ($(VENV_DIR)) Répertoire de l'environnement virtuel."
	@echo "  REQUIREMENTS : ($(REQUIREMENTS)) Fichier des dépendances."
	@echo "  ARGS         : Arguments passés à la commande 'make run' (ex: make run ARGS=\"--help\")."