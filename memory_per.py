import numpy as np

from sum_tree import SumTree


class MemoryPER:
    def __init__(self, size, epsilon, alpha, beta, beta_increment, abs_err_upper):
        self.tree = SumTree(size)

        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.abs_err_upper = abs_err_upper

    def add(self, state, action, reward, next_state, done):
        # Potentially inefficient search for maximum priority
        max_prio = np.max(self.tree.tree[-self.tree.size:])
        if max_prio == 0:
            max_prio = self.abs_err_upper
        # Add new transition with maximum priority to ensure it gets sampled at least once
        self.tree.add(max_prio, (state, action, reward, next_state, done))

    def sample(self, batch_size):
        pass
