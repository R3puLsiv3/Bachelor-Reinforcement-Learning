import numpy as np

from memory import Memory
from dqn import DQN


class Agent:
    def __init__(self, state_size, action_size, batch_size, memory_size, epsilon, min_epsilon, epsilon_decay, discount,
                 target_update):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.discount = discount
        self.target_update = target_update

        self.dqn = DQN(self.state_size, self.action_size)
        self.memory = Memory(self.memory_size)

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            state = np.reshape(state, (1, 2))
            return np.argmax(self.dqn.query_main(state))
        else:
            return np.random.randint(0, self.action_size)

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.min_epsilon, self.epsilon)

    def train(self):
        if self.memory.len() < self.batch_size:
            return

        self.dqn.target_main_delta += 1
        batch = self.memory.sample(self.batch_size)

        x = [memory[0] for memory in batch]
        y = self.dqn.query_main(x)
        target_qs = self.dqn.query_target([memory[3] for memory in batch])
        for index, (state, action, reward, next_state, done) in enumerate(batch):
            if not done:
                target_q = np.max(target_qs[index])
                new_q = reward + self.discount * target_q
            else:
                new_q = reward

            y = np.array(y)
            y[index][action] = new_q
        self.dqn.fit_main(x, y)

        if self.dqn.target_main_delta % self.target_update == 0:
            self.dqn.update_target()
