import random

from collections import deque


class Memory:
    def __init__(self, size):
        self.size = size
        self.replay_buffer = deque(maxlen=self.size)

    def add_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_memories(self, batch_size):
        return random.sample(self.replay_buffer, batch_size)
