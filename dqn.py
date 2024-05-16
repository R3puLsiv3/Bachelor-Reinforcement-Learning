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

        self.target_main_delta = 0

    def create_model(self):
        model = Sequential()

        model.add(Dense(16, input_dim=self.state_dim, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(self.action_dim, activation="linear"))

        model.compile(optimizer="adam", loss="mean_squared_loss")

        return model

    def query_main(self, states):
        self.main_model.predict(np.array(states))

    def query_target(self, states):
        self.target_model.predict(np.array(states))

    def update_target(self):
        self.target_model.set_weights(self.main_model.get_weights())
        self.target_main_delta = 0

    def fit_main(self, X, y, batch_size):
        self.main_model.fit(np.array(X), np.array(y), batch_size=batch_size, verbose=0, shuffle=False)
