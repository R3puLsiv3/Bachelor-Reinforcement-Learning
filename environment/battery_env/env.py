import gymnasium as gym
import numpy as np
import pandas as pd
import os
import random

from gymnasium import spaces


class EnvBase(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(21)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float64)

        self.actions = {0: -1., 1: -0.9, 2: -0.8, 3: -0.7, 4: -0.6, 5: -0.5, 6: -0.4, 7: -0.3, 8: -0.2, 9: -0.1, 10: 0.,
                        11: 0.1, 12: 0.2, 13: 0.3, 14: 0.4, 15: 0.5, 16: 0.6, 17: 0.7, 18: 0.8, 19: 0.9, 20: 1.}

        self.min_capacity = 0.
        self.max_capacity = 1.
        self.soc = random.uniform(0.0, 1.0)
        self.capacity = 50

        self.data_pointer = 0
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.data = pd.read_csv(dir_path + "/data/Data_Base_Env_2021.csv")
        self.data_length = self.data.shape[0] - 300
        self.total_demand = self.data["Total Demand [kW]"]
        self.pv_gen = self.data["PV Gen [kW]"]
        self.day_ahead_price = self.data["Day-ahead Price [EUR/kWh]"]

    def calculate_reward(self, old_soc):
        soc_delta = self.soc - old_soc
        battery_supply = soc_delta * self.capacity
        grid_demand = -self.total_demand[self.data_pointer] - self.pv_gen[self.data_pointer]
        GRID_demand = grid_demand + battery_supply

        if GRID_demand < 0:
            multiplier = 0.5
        else:
            multiplier = -1.
        return min(self.day_ahead_price[self.data_pointer] * GRID_demand * multiplier, 0)

    def step(self, action):
        self.data_pointer += 1
        done = self.data_pointer == self.data_length

        action = self.actions[action]

        old_soc = self.soc
        self.calculate_new_soc(action)

        reward = self.calculate_reward(old_soc)

        info = {}

        return np.asarray([self.total_demand[self.data_pointer], self.day_ahead_price[self.data_pointer], self.soc]), reward, done, False, info

    def calculate_new_soc(self, action):
        if action < 0.:
            self.soc = max((self.soc + action), self.min_capacity)
        elif action > 0.:
            self.soc = min((self.soc + action), self.max_capacity)

    def reset(self, seed=None, options=None):
        self.data_pointer = 0
        self.soc = random.uniform(0.0, 1.0)
        return np.asarray([self.total_demand[self.data_pointer], self.day_ahead_price[self.data_pointer], self.soc]), {}

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

        return np.asarray([self.total_demand[self.data_pointer], self.day_ahead_price[self.data_pointer], self.soc, self.timestamp[self.data_pointer]]), reward, done, False, info

    def reset(self, seed=None, options=None):
        self.data_pointer = 0
        self.soc = 1.
        return np.asarray([self.total_demand[self.data_pointer], self.day_ahead_price[self.data_pointer], self.soc, self.timestamp[self.data_pointer]]), {}


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

        return np.asarray([self.total_demand[self.data_pointer], self.day_ahead_price[self.data_pointer], self.soc, self.radiation[self.data_pointer]]), reward, done, False, info

    def reset(self, seed=None, options=None):
        self.data_pointer = 0
        self.soc = 1.
        return np.asarray([self.total_demand[self.data_pointer], self.day_ahead_price[self.data_pointer], self.soc, self.radiation[self.data_pointer]]), {}


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

        return np.asarray([self.total_demand[self.data_pointer], self.day_ahead_price[self.data_pointer], self.soc, self.timestamp[self.data_pointer], self.radiation[self.data_pointer]]), reward, done, False, info

    def reset(self, seed=None, options=None):
        self.data_pointer = 0
        self.soc = 1.
        return np.asarray([self.total_demand[self.data_pointer], self.day_ahead_price[self.data_pointer], self.soc, self.timestamp[self.data_pointer], self.radiation[self.data_pointer]]), {}
