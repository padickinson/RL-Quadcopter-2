import pickle
import numpy as np
from keras.models import load_model, save_model
from agents.agent import Agent
from agents.replay_buffer import ReplayBuffer
from agents.ornstein_uhlenbeck_noise import OUNoise
from agents.ddpg_v5.ddpg_actor import Actor
from agents.ddpg_v5.ddpg_critic import Critic

class DDPG(Agent):
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, env):
        # Changed this to use generic env instead of Task
        super().__init__(env)

        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 1e-2 # for soft update of target parameters

        # Critic Params
        self.critic_lr = 1e-3
        self.critic_decay = 1e-2

        # Actor Params
        self.actor_lr = 1e-4
        self.actor_decay = 0

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high, self.actor_lr, self.actor_decay)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high, self.actor_lr, self.actor_decay)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size, self.critic_lr, self.critic_decay)
        self.critic_target = Critic(self.state_size, self.action_size, self.critic_lr, self.critic_decay)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)





    def reset_episode(self):
        self.noise.reset()
        state = self.env.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done, training=True):
        # Since DDPG is an off-policy learner, add a training flag

        # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if training and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            self.steps_trained += 1

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state, training=True):
        # Add a training flag to decide whether to explore
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        if training:
            return list(action + self.noise.sample())  # add some noise for exploration
        else:
            return list(action)

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

    def save_model(self, filename):
        al = self.actor_local
        at = self.actor_target
        cl = self.critic_local
        ct = self.critic_target

        self.actor_local = None
        self.actor_target = None
        self.critic_local = None
        self.critic_target = None

        with open(filename+'.ddpg_agent') as f:
            pickle.dump(self, f)

        al.save(filename+'.actor_local')
        at.save(filename+'.actor_target')
        cl.save(filename+'.critic_local')
        ct.save(filename+'.critic_target')

        self.actor_local = al
        self.actor_target = at
        self.critic_local = cl
        self.critic_target = ct

    @classmethod
    def load_model(cls, filename):
        with open(filename+ '.ddpg_agent') as f:
            m = pickle.load(f)
        m.actor_local = load_model(filename+'.actor_local')
        m.actor_target = load_model(filename+'.actor_target')
        m.critic_local = load_model(filename+'.critic_local')
        m.critic_target = load_model(filename+'.critic_target')
        return m
