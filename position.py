import gymnasium
import numpy as np
import ale_py

class MontezumaEnvironment:
    def __init__(self, render_mode="human"):
        self.env = gymnasium.make("ALE/MontezumaRevenge-v5", render_mode=render_mode)
        self.ale = self.env.unwrapped.ale
        self.previous_lives = self.get_lives()

    def get_agent_position(self):
        # Obtenir la RAM du jeu
        ram = self.ale.getRAM()

        # Position X : RAM[42]
        # Position Y : RAM[43]
        x_pos = ram[42]
        y_pos = ram[43]

        return x_pos, y_pos

    def set_agent_position(self, x, y):
        # Pour modifier la RAM, nous devons le faire adresse par adresse
        self.ale.setRAM(42, x)  # Position X
        self.ale.setRAM(43, y)  # Position Y

    def get_lives(self):
        ram = self.ale.getRAM()
        return ram[58]

    def get_action_meanings(self, action):
        return self.env.unwrapped.get_action_meanings()[action]

    def run_game(self, num_steps=1000):
        for action_id, action_name in enumerate(self.env.unwrapped.get_action_meanings()):
            print(f"Action {action_id}: {action_name}")

        observation, info = self.env.reset()
        total_reward = 0
        step = 0

        self.set_agent_position(105, 148)

        while step < num_steps:
            # Action aléatoire
            action = self.env.action_space.sample()
            print(f"Action: {action, self.get_action_meanings(action)}")

            # Effectuer l'action
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            current_lives = self.get_lives()
            if current_lives < self.previous_lives:
                print(f"Lost a life! Lives remaining: {current_lives}")
                terminated = True

            self.previous_lives = current_lives

            # Obtenir et afficher la position
            x, y = self.get_agent_position()
            print(f"Step {step} - Position: X={x}, Y={y} - Reward: {reward}")

            step += 1

            if terminated or truncated:
                print("Episode finished")
                observation, info = self.env.reset()
                self.previous_lives = self.get_lives()
                self.set_agent_position(105, 148)

    def close(self):
        self.env.close()

def main():
    # Créer et exécuter l'environnement
    game = MontezumaEnvironment(render_mode="human")
    try:
        game.run_game(num_steps=500)  # Exécuter pendant 500 steps
    finally:
        game.close()

if __name__ == "__main__":
    main()

# Created/Modified files during execution:
# Aucun fichier n'est créé pendant l'exécution