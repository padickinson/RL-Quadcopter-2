import keras.backend as K
from keras import optimizers


class Critic(object):

    def __init__(self, model, lr=1e-3, decay=1e-2):
        self.model = model
        self.opt = optimizers.Adam(lr=lr, decay=decay)
        self.model.compile(self.opt, loss='mse')

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def step(self, states, actions, ys):
        self.model.train_on_batch([states, actions], ys)

    def __call__(self, states, actions):
        return self.model.predict_on_batch([states, actions])
