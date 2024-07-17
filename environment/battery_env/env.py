import gymnasium as gym
import numpy as np
import pandas as pd
import os

from gymnasium import spaces


class EnvBase(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(21)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(3,), dtype=np.float32)

        self.actions = {0: -1., 1: -0.9, 2: -0.8, 3: -0.7, 4: -0.6, 5: -0.5, 6: -0.4, 7: -0.3, 8: -0.2, 9: -0.1, 10: 0.,
                        11: 0.1, 12: 0.2, 13: 0.3, 14: 0.4, 15: 0.5, 16: 0.6, 17: 0.7, 18: 0.8, 19: 0.9, 20: 1.}

        self.min_capacity = 0.
        self.max_capacity = 1.
        self.soc = 1.
        self.capacity = 50_000

        self.data_pointer = 0
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.data = pd.read_csv(dir_path + "/data/hourly_data.csv")
        self.data_length = self.data.shape[0]
        self.demand_grid = self.data["demand_grid"]
        self.day_ahead_price = self.data["day_ahead_price"]

    def calculate_reward(self, old_soc):
        soc_delta = self.soc - old_soc
        battery_supply = soc_delta * self.capacity
        demand_GRID = self.demand_grid[self.data_pointer] - battery_supply

        if demand_GRID < 0:
            multiplier = -0.5
        else:
            multiplier = -1.
        return -(self.day_ahead_price[self.data_pointer] * demand_GRID * multiplier)

    def step(self, action):
        self.data_pointer += 1
        done = self.data_pointer == self.data_length

        action = self.actions[action]

        old_soc = self.soc
        self.calculate_new_soc(action)

        reward = self.calculate_reward(old_soc)

        info = {}

        return np.asarray([self.demand_grid[self.data_pointer], self.day_ahead_price[self.data_pointer], self.soc]), reward, done, False, info

    def calculate_new_soc(self, action):
        if action < 0.:
            if self.soc + action < 0.:
                action = self.soc
            self.soc = max((self.soc - action), self.min_capacity)
        elif action > 0.:
            if self.soc + action > self.max_capacity:
                action = self.max_capacity - self.soc
            self.soc = min((self.soc + action), self.max_capacity)

    def reset(self, seed=None, options=None):
        self.data_pointer = 0
        self.soc = 1.
        return np.asarray([self.demand_grid[self.data_pointer], self.day_ahead_price[self.data_pointer], self.soc]), {}

    def render(self):
        pass

    def close(self):
        pass


class EnvTimestamp(EnvBase):
    def __init__(self):
        super().__init__()
        self.timestamp = self.data["Timestamp"]

    def step(self, action):
        self.data_pointer += 1
        done = self.data_pointer == self.data_length

        action = self.actions[action]

        old_soc = self.soc
        self.calculate_new_soc(action)

        reward = self.calculate_reward(old_soc)

        info = {}

        return np.asarray([self.demand_grid[self.data_pointer], self.day_ahead_price[self.data_pointer], self.soc, self.timestamp[self.data_pointer]]), reward, done, False, info

    def reset(self, seed=None, options=None):
        self.data_pointer = 0
        self.soc = 1.
        return np.asarray([self.demand_grid[self.data_pointer], self.day_ahead_price[self.data_pointer], self.soc, self.timestamp[self.data_pointer]]), {}


class EnvRadiation(EnvBase):
    def __init__(self):
        super().__init__()
        self.timestamp = self.data["Radiation"]

    def step(self, action):
        self.data_pointer += 1
        done = self.data_pointer == self.data_length

        action = self.actions[action]

        old_soc = self.soc
        self.calculate_new_soc(action)

        reward = self.calculate_reward(old_soc)

        info = {}

        return np.asarray([self.demand_grid[self.data_pointer], self.day_ahead_price[self.data_pointer], self.soc, self.radiation[self.data_pointer]]), reward, done, False, info

    def reset(self, seed=None, options=None):
        self.data_pointer = 0
        self.soc = 1.
        return np.asarray([self.demand_grid[self.data_pointer], self.day_ahead_price[self.data_pointer], self.soc, self.radiation[self.data_pointer]]), {}


class EnvTimestampRadiation(EnvBase):
    def __init__(self):
        super().__init__()
        self.timestamp = self.data["Timestamp"]
        self.radiation = self.data["Radiation"]

    def step(self, action):
        self.data_pointer += 1
        done = self.data_pointer == self.data_length

        action = self.actions[action]

        old_soc = self.soc
        self.calculate_new_soc(action)

        reward = self.calculate_reward(old_soc)

        info = {}

        return np.asarray([self.demand_grid[self.data_pointer], self.day_ahead_price[self.data_pointer], self.soc, self.timestamp[self.data_pointer], self.radiation[self.data_pointer]]), reward, done, False, info

    def reset(self, seed=None, options=None):
        self.data_pointer = 0
        self.soc = 1.
        return np.asarray([self.demand_grid[self.data_pointer], self.day_ahead_price[self.data_pointer], self.soc, self.timestamp[self.data_pointer], self.radiation[self.data_pointer]]), {}
