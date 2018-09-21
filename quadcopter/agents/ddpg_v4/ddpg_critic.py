from keras import layers, models, optimizers
from keras import backend as K
from keras.initializers import RandomUniform, VarianceScaling



class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, lr, decay):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.decay = decay

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        # states = layers.BatchNormalization()(states)
        actions = layers.Input(shape=(self.action_size,), name='actions')
        # actions = layers.BatchNormalization()(actions)

        # Add hidden layer(s) for state pathway
        w_init = VarianceScaling(scale=1./3, mode='fan_in', distribution='uniform')
        net_states = layers.Dense(units=400, kernel_initializer=w_init, bias_initializer=w_init, activation='relu')(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.5)(net_states)

        concat = layers.concatenate([net_states, actions])

        net = layers.Dense(units=300, kernel_initializer=w_init, bias_initializer=w_init, activation='relu')(concat)
        net = layers.BatchNormalization()(net)
        net = layers.Dropout(0.5)(net)

        # Add final output layer to prduce action values (Q values)
        w_init = RandomUniform(-3e-3, 3e-3)
        Q_values = layers.Dense(units=1,
            kernel_initializer=w_init,
            bias_initializer=w_init,
            name='q_values',
            activation='linear')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.lr, decay=self.decay)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
