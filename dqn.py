import tensorflow as tf

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense


class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.main_model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.main_model.get_weights())

        self.target_main_delta = 0

    def create_model(self):
        model = Sequential()

        model.add(Input(shape=(self.state_size, )))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))

        model.compile(optimizer="adam", loss="mean_squared_error")

        return model

    def query_main(self, states):
        return self.main_model(tf.convert_to_tensor(states), training=False)

    def query_target(self, states):
        return self.target_model(tf.convert_to_tensor(states), training=False)

    def update_target(self):
        self.target_model.set_weights(self.main_model.get_weights())
        self.target_main_delta = 0

    def fit_main(self, x, y):
        self.main_model.train_on_batch(tf.convert_to_tensor(x), tf.convert_to_tensor(y))
