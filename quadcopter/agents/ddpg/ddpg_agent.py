import numpy as np
from keras.models import clone_model
import keras.backend as K
from agents.agent_new import Agent
from agents.replay_buffer import ReplayBuffer
from agents.ornstein_uhlenbeck_noise import OUNoise
from agents.ddpg.actor import Actor
from agents.ddpg.critic import Critic

class DDPGAgent(Agent):
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self,
                actor_model, tgt_actor_model,
                critic_model, tgt_critic_model,
                action_limits,
                actor_lr = 1e-4,
                critic_lr = 1e-3,
                critic_decay = 1e-2,
                tau=1e-3,
                gamma=0.99,
                process = None,
                rb_size = 1e6,
                minibatch_size = 64,
                warmup_episodes=0,
                episodes_trained=0,
                train_scores = None,
                test_scores = None,
                best_train_score = -np.inf
                ):
        # Changed this to use generic env instead of Task
        super().__init__(warmup_episodes,episodes_trained,train_scores,test_scores,best_train_score)
        self.actor = Actor(actor_model,critic_model,lr=actor_lr)
        self.tgt_actor = Actor(tgt_actor_model, tgt_critic_model, lr=actor_lr)
        self.tgt_actor.set_weights(self.actor.get_weights())

        self.critic = Critic(critic_model, lr=critic_lr, decay=critic_decay)
        self.tgt_critic = Critic(tgt_critic_model, lr=critic_lr, decay=critic_decay)
        self.tgt_critic.set_weights(self.critic.get_weights())

        self.action_limits = action_limits
        self.process = process
        self.minibatch_size = minibatch_size
        self.buffer = ReplayBuffer(int(rb_size), self.minibatch_size)
        self.tau = tau
        self.gamma = gamma

        self.state_space = K.int_shape(critic_model.inputs[0])[1]
        self.action_space = K.int_shape(critic_model.inputs[1])[1]

        self.learning_phase = 1
        if process is None:
            self.process = OUNoise(size=self.action_space, theta=0.15, mu=0,
                                             sigma=0.2)
        else:
            self.process = process

    def sense(self, s, a, r, s_new, done):
        s = np.reshape(s, [-1, self.state_space])
        s_new = np.reshape(s_new, [-1, self.state_space])
        self.buffer.add(s, a, r, s_new, done)

    def act(self, s):
        s = np.reshape(s, [-1, self.state_space])
        a = self.tgt_actor(s)
        # Cache.
        self.last_state = np.copy(s)
        self.last_action = np.copy(a)
        if self.learning_phase:
            a += self.process.sample()
        a = np.clip(a, self.action_limits[0], self.action_limits[1])

        self.last_action_noisy = np.copy(a)
        return a

    def new_episode(self):
        self.process.reset()

    def train_step(self):
        if len(self.buffer.memory) < self.minibatch_size:
            return

        minibatch = self.buffer.sample(self.minibatch_size)
        states = np.zeros([len(minibatch), self.state_space])
        states_new = np.zeros([len(minibatch), self.state_space])
        actions = np.zeros([len(minibatch), self.action_space])
        r = np.zeros([len(minibatch), 1])
        dones = np.zeros([len(minibatch), 1])

        for i in range(len(minibatch)):
            states[i], actions[i], r[i], states_new[i], dones[i] = minibatch[i]

        # Estimate Q_values
        critic_out = self.critic(states_new, self.actor(states_new))
        tgt_critic_out = self.tgt_critic(states_new, self.tgt_actor(states_new))

        # Q-values using tgt_critic
        ys = r + self.gamma * tgt_critic_out

        # Train local critic and actor
        self.critic.step(states, actions, ys)
        self.actor.step(states)

        # Soft weight updates for target critic and actor
        critic_weights = self.critic.get_weights()
        tgt_critic_weights = self.tgt_critic.get_weights()
        for i in range(len(critic_weights)):
            tgt_critic_weights[i] = (1 - self.tau) * tgt_critic_weights[i] + \
                self.tau * critic_weights[i]
        self.tgt_critic.set_weights(tgt_critic_weights)

        actor_weights = self.actor.get_weights()
        tgt_actor_weights = self.tgt_actor.get_weights()
        for i in range(len(actor_weights)):
            tgt_actor_weights[i] = (1 - self.tau) * tgt_actor_weights[i] + \
                self.tau * actor_weights[i]
        self.tgt_actor.set_weights(tgt_actor_weights)
