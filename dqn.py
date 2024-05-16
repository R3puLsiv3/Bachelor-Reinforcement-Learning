import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class DQN:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.main_model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.main_model.get_weights())

        self.target_update_delta = 0

    def create_model(self):
        model = Sequential()

        model.add(Dense(16, input_dim=self.state_dim, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(self.action_dim, activation="linear"))

        model.compile(optimizer="adam", loss="mean_squared_loss")

        return model

    def query(self, state):
        self.main_model.predict(np.array(state))
