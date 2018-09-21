from keras import layers, models, optimizers
from keras import backend as K



class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, lr):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr

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
        net_states = layers.Dense(units=32, activation='relu')(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.5)(net_states)
        net_states = layers.Dense(units=64, activation='relu')(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.5)(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=32, activation='relu')(actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_states = layers.Dropout(0.5)(net_states)
        net_actions = layers.Dense(units=64, activation='relu')(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_states = layers.Dropout(0.5)(net_states)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways

        # Changes here:
        # 1. Change Add to concatenate
        # 2. Remove the relu activation (redundant)
        # 3. Add Batch Normalization

        # net = layers.Add()([net_states, net_actions])
        # net = layers.Activation('relu')(net)

        net = layers.concatenate([net_states, net_actions])
        net = layers.BatchNormalization()(net)

        # Add more layers to the combined network if needed
        # 4. Add a Dense layer
        net = layers.Dense(units=128, activation='relu')(net)
        net = layers.BatchNormalization()(net)
        net = layers.Dropout(0.5)(net)


        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.lr)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
