import keras.backend as K
from keras import optimizers


class Actor(object):
    """Actor model for Actor-Critic
    """
    def __init__(self, model, critic_model, lr=1e-4):
        self.model = model
        self.opt = optimizers.Adam(lr)
        self.model.compile(self.opt, 'mse')

        # Critic input tensors are (state_input, action_input)
        state_input = self.model.inputs[0]

        # Get Q-values from critic model (using actor outputs as action_input)
        critic_out = critic_model([state_input, self.model(state_input)])

        # Loss is given by Q value action gradients
        loss = -K.mean(critic_out)

        # Get the weights up to update
        updates = self.opt.get_updates(
            params=self.model.trainable_weights,
            loss=loss)
        updates += self.model.updates
        # learning_phase added in case of batchnorm layers.
        self.train_step = K.function(
            inputs=[state_input, K.learning_phase()], outputs=[], updates=updates)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def step(self, states):
        learning_phase = 1  # Signalize training phase. Handled implicitly for prediction.
        self.train_step([states, learning_phase])

    def __call__(self, state):
        return self.model.predict_on_batch([state])
