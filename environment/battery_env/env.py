import gymnasium as gym
import numpy as np
import pandas as pd
import os

from gymnasium import spaces


class Env(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(21)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(3,), dtype=np.float32)

        self.actions = {0: -1., 1: -0.9, 2: -0.8, 3: -0.7, 4: -0.6, 5: -0.5, 6: -0.4, 7: -0.3, 8: -0.2, 9: -0.1, 10: 0.,
                        11: 0.1, 12: 0.2, 13: 0.3, 14: 0.4, 15: 0.5, 16: 0.6, 17: 0.7, 18: 0.8, 19: 0.9, 20: 1.}

        self.min_capacity = 0.
        self.max_capacity = 1.
        self.charge_efficiency = 0.7
        self.discharge_efficiency = 0.7
        self.soc = 1.

        self.data_pointer = 0
        self.data_length = 1400
        dir_path = os.path.dirname(os.path.realpath(__file__))
        demands = pd.read_csv(dir_path + "/data/demand.csv")
        demands["Total_demand"] = demands.iloc[1:4].sum(axis=1)
        self.demand = demands["Total_demand"]
        self.day_ahead_price = pd.read_csv(dir_path + "/data/day_ahead_price", skiprows=lambda x: x % 2)["Price"]

    def calculate_reward(self, action):
        # Return the negative of grid usage times price?
        pass

    def step(self, action):
        self.data_pointer += 1
        done = self.data_pointer == self.data_length

        action = self.actions[action]

        action = self.calculate_new_soc(action)

        reward = self.calculate_reward(action)

        info = {}

        return np.asarray([self.demand[self.data_pointer], self.day_ahead_price[self.data_pointer], self.soc]), reward, done, False, info

    def calculate_new_soc(self, action):
        # Should Charge/Discharge rate be included in these calculations?
        if action < 0.:
            if self.soc - action < 0.:
                action = self.soc
            self.soc = max((self.soc - action), self.min_capacity)
        elif action > 0.:
            if self.soc + action > self.max_capacity:
                action = self.max_capacity - self.soc
            self.soc = min((self.soc + action), self.max_capacity)
        return action

    def reset(self, seed=None, options=None):
        self.data_pointer = 0
        self.soc = 1.
        return np.asarray([self.demand[self.data_pointer], self.day_ahead_price[self.data_pointer], self.soc]), {}

    def render(self):
        pass

    def close(self):
        pass
