import numpy as np

from memory import Memory
from dqn import DQN


class Agent:
    def __init__(self, state_dim, action_dim, memory_size, epsilon, batch_size, discount, target_update):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_size = memory_size
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.discount = discount
        self.target_update = target_update

        self.dqn = DQN(self.state_dim, self.action_dim)
        self.memory = Memory(self.memory_size)

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            return np.argmax(self.dqn.query_main(state))
        else:
            return np.random.randint(0, self.action_dim)

    def train(self, episode_end):
        if self.memory.len() < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)

        X = [memory[0] for memory in batch]
        y = self.dqn.query_main(X)

        target_qs = self.dqn.query_target([memory[3] for memory in batch])

        for index, (state, action, reward, next_state, done) in enumerate(batch):
            if not done:
                target_q = np.max(target_qs[index])
                new_q = reward + self.discount * target_q
            else:
                new_q = reward

            y[index][action] = new_q

        self.dqn.fit_main(X, y, self.batch_size)

        if episode_end:
            self.dqn.target_main_delta += 1

        if self.dqn.target_main_delta % self.target_update == 0:
            self.dqn.update_target()
