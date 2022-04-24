# python 3.8 works!
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class NNet():
    def __init__(self, model=None):
        if model is None:
            self.model = self.build_model()
        else:
            self.model = model
        return

    def build_model(self):
        inputs = keras.Input(shape=(10,), name="board_pos")
        d_1 = layers.Dense(5, activation='relu', name='dense_1')(inputs)
        d_2 = layers.Dense(7, activation='relu', name='dense_2')(d_1)

        policy = layers.Dense(9, activation='softmax', name='policy')(d_2)
        eval = layers.Dense(1, activation='tanh', name='evaluation')(d_2)

        return keras.Model(inputs=inputs, outputs=[policy, eval])

    # state is a np.array => [[d1], [d2], ...]
    def predict(self, input):
        policy, eval = self.model(input)
        return policy[0].numpy(), eval.numpy()[0][0]

    # Input Data Format
    # np.array( [[10 inputs], [10 inputs]], ndim=2)
    # Output Data Format
    # [np.array([[10 possible moves], [10 possible moves]]), np.array([[pi], [pi]])]
    def train(self, train_x, train_y, lr=0.1, epochs=25):
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=keras.optimizers.Adam(learning_rate=lr))
        self.model.fit(train_x, train_y, epochs=  epochs, verbose = 0)
    
    def clone(self):
        new_model = keras.models.clone_model(self.model)
        new_model.set_weights(self.model.get_weights())
        return NNet(new_model)




################
##NETWORK INFO##
################

# Input Data Format
# np.array( [[10 inputs], [10 inputs]], ndim=2)

# Output Data
# [np.array([[10 possible moves], [10 possible moves]]), np.array([[pi], [pi]])]

###########
##EXAMPLE##
###########
# model = NNet()
# inputs = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0, -1], [-1, 0, 0, 0, 0, 0, 0, 1, 0, 1], [-1, 0, 1, 0, 0, 0, 0, 1, 0, -1], [-1, 0, 1, 0, 0, 0, 0, 1, -1, 1], [-1, 0, 1, 0, 1, 0, 0, 1, -1, -1], [-1, 0, 1, 0, 1, 0, -1, 1, -1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0, -1], [0, 0, -1, 0, 0, 0, 0, 1, 0, 1], [0, 0, -1, 0, 0, 0, 1, 1, 0, -1], [0, -1, -1, 0, 0, 0, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0, -1], [-1, 0, 1, 0, 0, 0, 0, 0, 0, 1], [-1, 0, 1, 0, 0, 0, 0, 1, 0, -1], [-1, 0, 1, 0, 0, -1, 0, 1, 0, 1], [-1, 0, 1, 1, 0, -1, 0, 1, 0, -1], [-1, 0, 
# 1, 1, 0, -1, -1, 1, 0, 1], [-1, 0, 1, 1, 0, -1, -1, 1, 1, -1], [-1, -1, 1, 1, 0, -1, -1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0, -1], [-1, 0, 1, 0, 0, 0, 0, 0, 0, 1], [-1, 0, 1, 0, 1, 0, 0, 0, 0, -1], [-1, 0, 1, 0, 1, 0, 0, -1, 0, 1], [-1, 1, 1, 0, 1, 0, 0, -1, 0, -1], [-1, 1, 1, 0, 1, -1, 0, -1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, -1], [-1, 0, 
# 0, 0, 0, 0, 0, 0, 1, 1], [-1, 0, 0, 0, 1, 0, 0, 0, 1, -1], [-1, 0, 0, 0, 1, -1, 0, 0, 1, 1], [-1, 0, 0, 0, 1, -1, 1, 0, 1, -1], [-1, 0, -1, 0, 1, -1, 1, 0, 1, 1], [-1, 1, -1, 0, 1, -1, 1, 0, 1, -1], [-1, 
# 1, -1, 0, 1, -1, 1, -1, 1, 1]]

# policies = [[0.25, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25], [0.2, 0.2, 0.2, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2], [0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.0, 0.2, 0.2, 0.2, 0.0, 0.2], [0.0, 0.2, 
# 0.0, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0], [0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.4, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.25, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25], [0.2, 0.2, 0.2, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2], [0.2, 0.2, 0.0, 0.0, 0.2, 0.2, 0.2, 0.0, 0.0], [0.2, 0.2, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.2], [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8], [0.25, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25], [0.2, 0.2, 0.0, 0.0, 0.2, 0.2, 0.0, 0.2, 0.0], [0.0, 0.2, 0.0, 0.0, 0.2, 0.0, 0.2, 0.2, 0.2], [0.0, 0.2, 0.0, 0.0, 0.2, 0.2, 0.2, 0.0, 0.2], [0.0, 0.2, 0.0, 0.2, 0.2, 0.0, 0.2, 0.0, 0.2], [0.0, 0.2, 0.0, 0.0, 0.4, 0.0, 0.2, 0.0, 0.2], [0.0, 0.2, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.2], [0.0, 0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.25, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25], [0.2, 0.2, 0.0, 0.0, 0.2, 0.2, 0.0, 0.2, 0.0], [0.0, 0.2, 0.0, 0.0, 0.2, 0.0, 0.2, 0.2, 0.2], [0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.2, 0.2, 0.0], [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0], [0.0, 0.0, 0.0, 0.2, 0.0, 0.4, 0.2, 0.0, 0.2], [0.0, 0.0, 0.0, 0.16666666666666666, 0.0, 0.0, 0.8333333333333334, 0.0, 0.0], [0.25, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25], [0.2, 0.2, 0.2, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0], [0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0], [0.0, 0.2, 0.2, 0.0, 0.0, 0.2, 0.2, 0.2, 0.0], [0.0, 0.2, 0.2, 0.2, 0.0, 0.0, 0.2, 0.2, 0.0], [0.0, 0.2, 0.4, 0.2, 0.0, 0.0, 0.0, 0.2, 0.0], [0.0, 0.16666666666666666, 0.0, 0.16666666666666666, 0.0, 0.0, 0.0, 0.6666666666666666, 0.0], [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.8, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

# rewards = [1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# train_x = np.array(inputs, dtype=np.float32)
# print("train_x shape: ",train_x.shape)
# print()

# policies = np.array(policies, dtype=np.float32)
# print("policies shape: ", policies.shape)
# print()

# rewards = np.array(rewards, dtype=np.float32)
# rewards = np.reshape(rewards, (len(rewards), 1))
# print("rewards shape: ", rewards.shape)
# print()

# train_y = [policies, rewards]

# model.train(train_x, train_y)

# print(model.predict(np.array([train_x[0]])))

