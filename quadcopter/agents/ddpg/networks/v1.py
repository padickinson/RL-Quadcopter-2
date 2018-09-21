import keras.backend as K
from keras.initializers import RandomUniform, VarianceScaling
from keras.layers import Input, Dense, concatenate, Lambda, Add, Activation, BatchNormalization
from keras.models import Model
from keras import optimizers

def create_actor(n_states, n_actions, action_range, action_low):
    state_input = Input(shape=(n_states,))

    h = Dense(32, activation='relu')(state_input)
    h = BatchNormalization()(h)
    h = Dense(64, activation='relu')(h)
    h = BatchNormalization()(h)
    h = Dense(32, activation='relu')(h)
    h = BatchNormalization()(h)

    raw_actions = Dense(n_actions,  activation='sigmoid')(h)
    actions = Lambda(lambda x: (x * action_range) + action_low,
        name='actions')(raw_actions)
    return Model(inputs=[state_input], outputs=[actions])


def create_critic(n_states, n_actions):
    states = Input(shape=(n_states,), name='states')
    actions = Input(shape=(n_actions,), name='actions')

    # Add hidden layer(s) for state pathway
    net_states = Dense(units=32, activation='relu')(states)
    net_states = BatchNormalization()(net_states)
    net_states = Dense(units=64, activation='relu')(net_states)
    net_states = BatchNormalization()(net_states)

    # Add hidden layer(s) for action pathway
    net_actions = Dense(units=32, activation='relu')(actions)
    net_actions = BatchNormalization()(net_actions)
    net_actions = Dense(units=64, activation='relu')(net_actions)
    net_actions = BatchNormalization()(net_actions)
    # Try different layer sizes, activations, add batch normalization, regularizers, etc.

    # Combine state and action pathways
    net = Add()([net_states, net_actions])
    net = Activation('relu')(net)

    # Add more layers to the combined network if needed

    # Add final output layer to prduce action values (Q values)
    Q_values = Dense(units=1, name='q_values', activation='linear')(net)

    # Create Keras model
    return Model(inputs=[states, actions], outputs=Q_values)
