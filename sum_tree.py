import numpy as np


class SumTree:
    # Adapted from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5
    # .2_Prioritized_Replay_DQN/RL_brain.py
    # and https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/tree.py

    def __init__(self, size):
        self.size = size
        self.tree_size = 2 * self.size - 1
        # Create binary tree with (size - 1) nodes and size leaves
        self.tree = np.zeros(self.tree_size)
        self.replay_buffer = np.zeros(size)

        # Keep track of position for adding new data
        self.data_pointer = 0

    def add(self, prio, transition):
        self.replay_buffer[self.data_pointer] = transition
        # Find corresponding position in tree and update tree
        tree_pointer = self.data_pointer + self.size - 1
        self.update(tree_pointer, prio)

        self.data_pointer = (self.data_pointer + 1) % self.size

    def update(self, tree_pointer, prio):
        old_prio = self.tree[tree_pointer]
        difference = prio - old_prio
        # Propagate priority difference through the tree
        while tree_pointer != 0:
            tree_pointer = (tree_pointer - 1) // 2
            self.tree[tree_pointer] += difference

    def retrieve(self, cum_sum):
        parent_pos = 0
        while True:
            child_left_pos = 2 * parent_pos + 1
            child_right_pos = child_left_pos + 1

            # In case leaf depth is reached
            if child_left_pos >= self.tree_size:
                leaf_pos = parent_pos
                break
            # Keep searching for higher priority nodes
            else:
                if cum_sum <= self.tree[child_left_pos]:
                    parent_pos = child_left_pos
                else:
                    cum_sum -= self.tree[child_left_pos]
                    parent_pos = child_right_pos

        data_pos = leaf_pos - self.size + 1
        return leaf_pos, self.tree[leaf_pos], self.replay_buffer[data_pos]

    def total_prio(self):
        return self.tree[0]





