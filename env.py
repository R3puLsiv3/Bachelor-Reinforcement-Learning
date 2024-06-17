import gymnasium as gym
import numpy as np

from gymnasium import spaces
from battery import Battery


class Env(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(21)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(3,), dtype=np.float32)

        self.actions = {0: -1., 1: -0.9, 2: -0.8, 3: -0.7, 4: -0.6, 5: -0.5, 6: -0.4, 7: -0.3, 8: -0.2, 9: -0.1, 10: 0.,
                        11: 0.1, 12: 0.2, 13: 0.3, 14: 0.4, 15: 0.5, 16: 0.6, 17: 0.7, 18: 0.8, 19: 0.9, 20: 1.}

        self.battery = Battery()

    def step(self, action):
        pass
