from collections import namedtuple, deque
import random
import torch

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity, device=None):
        self.memory = deque(maxlen=capacity)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        """Retourne le nombre d'expériences dans le buffer"""
        return len(self.memory)  # Ajout de cette méthode

    def push(self, state, action, reward, next_state, done):
        """Ajoute une expérience au buffer"""
        def process_tensor(x, dtype=torch.float32):
            if isinstance(x, torch.Tensor):
                x = x.detach()
            else:
                x = torch.tensor(x, dtype=dtype)

            # Convertir les scalaires en tenseurs 1D
            if x.dim() == 0:
                x = x.view(1)
            return x.cpu()

        # Conversion et stockage sur CPU
        state = process_tensor(state)
        next_state = process_tensor(next_state)
        action = process_tensor(action, dtype=torch.long)
        reward = process_tensor(reward)
        done = process_tensor(done, dtype=torch.float32)

        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Échantillonne un batch d'expériences"""
        experiences = random.sample(self.memory, batch_size)

        # Empile les tenseurs
        states = torch.stack([exp.state for exp in experiences]).to(self.device)
        actions = torch.cat([exp.action for exp in experiences]).to(self.device)
        rewards = torch.cat([exp.reward for exp in experiences]).to(self.device)
        next_states = torch.stack([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.cat([exp.done for exp in experiences]).to(self.device)

        return (states, actions, rewards, next_states, dones)