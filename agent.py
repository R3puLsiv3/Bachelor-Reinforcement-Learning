import numpy as np

from memory import Memory
from dqn import DQN


class Agent:
    def __init__(self, state_dim, action_dim, memory_size, epsilon):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_size = memory_size
        self.epsilon = epsilon

        self.network = DQN(self.state_dim, self.action_dim)
        self.memory = Memory(self.memory_size)

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            return np.argmax(self.network.query(state))
        else:
            return np.random.randint(0, self.action_dim)