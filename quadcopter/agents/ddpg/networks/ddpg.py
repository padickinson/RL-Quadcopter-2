import keras.backend as K
from keras.initializers import RandomUniform, VarianceScaling
from keras.layers import Input, Dense, concatenate, Lambda
from keras.models import Model
from keras import optimizers

def create_actor(n_states, n_actions):
    state_input = Input(shape=(n_states,))

    w_init = VarianceScaling(scale=1./3, mode='fan_in', distribution='uniform')
    h1 = Dense(400, kernel_initializer=w_init,
               bias_initializer=w_init, activation='relu')(state_input)
    h2 = Dense(300, kernel_initializer=w_init,
               bias_initializer=w_init, activation='relu')(h1)

    w_init = RandomUniform(-3e-3, 3e-3)
    out = Dense(n_actions, kernel_initializer=w_init,
                bias_initializer=w_init, activation='tanh')(h2)
    out = Lambda(lambda x: 2 * x, output_shape=(1,))(out)  # Since the output range is -2 to 2.

    return Model(inputs=[state_input], outputs=[out])


def create_critic(n_states, n_actions):
    state_input = Input(shape=(n_states,))
    action_input = Input(shape=(n_actions,))

    w_init = VarianceScaling(scale=1./3, mode='fan_in', distribution='uniform')
    h1 = Dense(400, kernel_initializer=w_init,
               bias_initializer=w_init, activation='relu')(state_input)
    x = concatenate([h1, action_input])
    h2 = Dense(300, kernel_initializer=w_init, bias_initializer=w_init, activation='relu')(x)

    w_init = RandomUniform(-3e-3, 3e-3)
    out = Dense(1, kernel_initializer=w_init, bias_initializer=w_init, activation='linear')(h2)

    return Model(inputs=[state_input, action_input], outputs=out)
