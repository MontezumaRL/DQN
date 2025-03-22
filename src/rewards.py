# fichier dqn/src/rewards.py
import math

class RewardSystem:
    def __init__(self):
        # Paramètres de base des récompenses
        self.time_penalty = -0.1
        self.life_loss_penalty = -10.0
        self.timeout_penalty = -10.0
        self.skeleton_reward = 50.0
        # Flag pour savoir si il a passé 1 fois le squelette
        self.flag_skeleton = False

        self.key_position = (21, 192)  # Position de la clé
        self.previous_distance = None
        # self.distance_reward_factor = 0.1  # Facteur multiplicateur pour la récompense de distance

    def reset(self):
        """Réinitialise les états des récompenses"""
        self.flag_skeleton = False
        self.previous_distance = None

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

    def calculate_distance_to_key(self, x, y):
        """Calcule la distance euclidienne entre l'agent et la clé"""
        return math.sqrt((x - self.key_position[0]) ** 2 + (y - self.key_position[1]) ** 2)

    def calculate_distance_reward(self, x, y):
        """Calcule la récompense basée sur la distance à la clé"""
        current_distance = self.calculate_distance_to_key(x, y)
        # print(f"current_distance: {current_distance}")
        # print(f"previous_distance: {self.previous_distance}")

        # Bonus si très proche de la clé (par exemple distance < 5)
        if current_distance < 5:
            return 20.0  # Bonus important

        # Si c'est la première fois qu'on calcule la distance
        if self.previous_distance is None:
            self.previous_distance = current_distance
            return 0.0

        # Calculer la différence de distance
        distance_difference = self.previous_distance - current_distance
        self.previous_distance = current_distance
        # print(f"distance_difference: {distance_difference}")

        # Récompense positive si on se rapproche, négative si on s'éloigne
        return distance_difference

    def calculate_total_reward(self, base_reward, x, y, life_lost, timeout):
        """Calcule la récompense totale en combinant toutes les récompenses"""
        total_reward = base_reward
        # print(f"base_reward: {base_reward}")

        # Ajouter la pénalité de temps
        total_reward += self.calculate_time_penalty()
        # print(f"time_penalty: {self.calculate_time_penalty()}")

        # Ajouter la récompense du bonbon
        total_reward += self.calculate_skeleton_reward(x, y)
        # print(f"skeleton_reward: {self.calculate_skeleton_reward(x, y)}")

        # Ajouter la récompense de distance
        total_reward += self.calculate_distance_reward(x, y)

        # Ajouter la pénalité de perte de vie
        if life_lost:
            total_reward += self.calculate_life_loss_penalty()
            # print(f"life_loss_penalty: {self.calculate_life_loss_penalty()}")

        # Ajouter la pénalité de timeout
        if timeout:
            total_reward += self.calculate_timeout_penalty()
            # print(f"timeout_penalty: {self.calculate_timeout_penalty()}")

        return total_reward