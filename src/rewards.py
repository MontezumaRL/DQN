import math

class RewardSystem:
    def __init__(self):
        # Récompenses principales
        self.key_reward = 100.0          # Récompense maximale pour atteindre la clé
        self.skeleton_reward = 50.0      # Récompense pour passer le squelette

        # Récompenses d'exploration
        self.leftward_reward = 5.0       # Récompense pour nouvelle position vers la gauche
        self.distance_reward_factor = 1.0 # Facteur pour la distance à la clé
        self.close_to_key_reward = 20.0  # Bonus quand très proche de la clé
        self.close_to_key_threshold = 5   # Distance considérée comme proche de la clé

        # Pénalités
        self.life_loss_penalty = -25.0   # Pénalité pour perte de vie
        self.timeout_penalty = -30.0     # Pénalité pour timeout

        # États et positions
        self.key_position = (21, 192)
        self.visited_positions = set()
        self.previous_position = None
        self.previous_distance = None
        self.flag_skeleton = False

    def reset(self):
        """Réinitialise les états des récompenses"""
        self.visited_positions.clear()
        self.previous_position = None
        self.previous_distance = None
        self.flag_skeleton = False

    def calculate_distance_to_key(self, x, y):
        """Calcule la distance euclidienne entre l'agent et la clé"""
        return math.sqrt((x - self.key_position[0])**2 + (y - self.key_position[1])**2)

    def calculate_distance_reward(self, x, y):
        """Calcule la récompense basée sur la distance à la clé"""
        current_distance = self.calculate_distance_to_key(x, y)

        # Bonus important si très proche de la clé
        if current_distance < self.close_to_key_threshold:
            return self.close_to_key_reward

        # Première mesure de distance
        if self.previous_distance is None:
            self.previous_distance = current_distance
            return 0.0

        # Calculer la progression vers la clé
        distance_difference = self.previous_distance - current_distance
        self.previous_distance = current_distance

        # Récompense proportionnelle à la progression
        return distance_difference * self.distance_reward_factor

    def calculate_leftward_exploration_reward(self, x, y):
        """Calcule la récompense pour l'exploration vers la gauche"""
        if (x, y) not in self.visited_positions:
            if self.previous_position is not None and x < self.previous_position[0]:
                self.visited_positions.add((x, y))
                self.previous_position = (x, y)
                return self.leftward_reward

        self.previous_position = (x, y)
        return 0.0

    def calculate_skeleton_reward(self, x, y):
        """Calcule la récompense pour le passage du squelette"""
        if x <= 39 and not self.flag_skeleton:
            self.flag_skeleton = True
            print("Premier passage du squelette !")
            return self.skeleton_reward
        return 0.0

    def calculate_total_reward(self, base_reward, x, y, life_lost, timeout):
        """
        Calcule la récompense totale avec une hiérarchie claire des récompenses
        """
        # Commencer avec la récompense de base du jeu
        total_reward = base_reward

        # 1. Récompenses principales (objectifs majeurs)
        skeleton_reward = self.calculate_skeleton_reward(x, y)
        total_reward += skeleton_reward

        # 2. Récompenses de progression (distance à la clé)
        distance_reward = self.calculate_distance_reward(x, y)
        total_reward += distance_reward

        # 3. Récompenses d'exploration
        exploration_reward = self.calculate_leftward_exploration_reward(x, y)
        total_reward += exploration_reward

        # 4. Pénalités
        if life_lost:
            total_reward += self.life_loss_penalty
        if timeout:
            total_reward += self.timeout_penalty

        # Debug logging (à commenter en production)
        # self._debug_rewards(base_reward, skeleton_reward, distance_reward,
        #                   exploration_reward, life_lost, timeout)

        return total_reward

    def _debug_rewards(self, base_reward, skeleton_reward, distance_reward,
                      exploration_reward, life_lost, timeout):
        """Affiche les détails des récompenses pour le debugging"""
        print("\nDétail des récompenses:")
        print(f"Base reward: {base_reward:>10.2f}")
        print(f"Skeleton: {skeleton_reward:>12.2f}")
        print(f"Distance: {distance_reward:>12.2f}")
        print(f"Exploration: {exploration_reward:>9.2f}")
        if life_lost:
            print(f"Life loss: {self.life_loss_penalty:>11.2f}")
        if timeout:
            print(f"Timeout: {self.timeout_penalty:>12.2f}")