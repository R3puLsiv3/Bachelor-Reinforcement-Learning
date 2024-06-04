import numpy as np
import random

from sum_tree import SumTree


class MemoryPER:
    # Adapted from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5
    # .2_Prioritized_Replay_DQN/RL_brain.py

    def __init__(self, size, epsilon, alpha, beta, beta_increment):
        self.tree = SumTree(size)

        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.abs_err_upper = 1.

    def add(self, state, action, reward, next_state, done):
        # Potentially inefficient search for maximum priority
        max_prio = np.max(self.tree.tree[-self.tree.size:])
        if max_prio == 0:
            max_prio = self.abs_err_upper
        # Add new transition with maximum priority to ensure it gets sampled at least once
        self.tree.add(max_prio, (state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch_pos, batch, importance_sampling_weights = (
            np.empty((batch_size,)), np.empty((batch_size, self.tree.replay_buffer[0].size)), np.empty((batch_size, 1)))
        prio_segment = self.tree.total_prio() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment])
        min_prob = np.min(self.tree.tree[-self.tree.size:])

        for i in range(batch_size):
            lower, upper = prio_segment * i, prio_segment * (i + 1)
            prio = np.random.uniform(lower, upper)
            pos, prio, data = self.tree.retrieve(prio)
            prob = prio / self.tree.total_prio()
            importance_sampling_weights[i, 0] = np.power(prob / min_prob, -self.beta)
            batch_pos[i], batch[i, :] = pos, data
        return batch_pos, batch, importance_sampling_weights

    def batch_update(self, tree_pos, abs_errors):
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        prios = np.power(clipped_errors, self.alpha)

        for pos, prio in zip(tree_pos, prios):
            self.tree.update(pos, prio)

