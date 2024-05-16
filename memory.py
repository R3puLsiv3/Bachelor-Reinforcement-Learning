import random

from collections import deque


class Memory:
    def __init__(self, size):
        self.size = size
        self.replay_buffer = deque(maxlen=self.size)

    def len(self):
        return len(self.replay_buffer)

    def add(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.replay_buffer, batch_size)
