# fichier dqn/src/rewards.py

class RewardSystem:
    def __init__(self):
        # Paramètres de base des récompenses
        self.time_penalty = -0.1
        self.life_loss_penalty = -10.0
        self.timeout_penalty = -10.0
        self.skeleton_reward = 50.0
        # Flag pour savoir si il a passé 1 fois le squelette
        self.flag_skeleton = False

    def reset(self):
        """Réinitialise les états des récompenses"""
        self.flag_skeleton = False

    def calculate_time_penalty(self):
        """Calcule la pénalité de temps"""
        return self.time_penalty

    def calculate_life_loss_penalty(self):
        """Calcule la pénalité de perte de vie"""
        return self.life_loss_penalty

    def calculate_timeout_penalty(self):
        """Calcule la pénalité de timeout"""
        return self.timeout_penalty

    def calculate_skeleton_reward(self, x, y):
        """Calcule la récompense si il passe le squelette"""
        # Si il passe le squelette alors qu'il ne l'avait pas encore passé
        if x <= 39 and not self.flag_skeleton:
            self.flag_skeleton = True
            print("1er passage du squelette")
            return self.skeleton_reward
        return 0.0

    def calculate_total_reward(self, base_reward, x, y, life_lost=False, timeout=False):
        """Calcule la récompense totale en combinant toutes les récompenses"""
        total_reward = base_reward

        # Ajouter la pénalité de temps
        total_reward += self.calculate_time_penalty()

        # Ajouter la récompense du bonbon
        total_reward += self.calculate_skeleton_reward(x, y)

        # Ajouter la pénalité de perte de vie
        if life_lost:
            total_reward += self.calculate_life_loss_penalty()

        # Ajouter la pénalité de timeout
        if timeout:
            total_reward += self.calculate_timeout_penalty()

        return total_reward