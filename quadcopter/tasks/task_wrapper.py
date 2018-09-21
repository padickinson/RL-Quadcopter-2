from gym.core import Env
from gym.spaces import Box
import numpy as np


class TaskWrapper(Env):
    def __init__(self,task):
        self.task = task
        self.action_space = Box(self.task.action_low,
            self.task.action_high,
            shape=(self.task.action_size,))
        self.observation_space = Box(
            -np.inf, np.inf, shape=(self.task.state_size,))
        super().__init__()

    def step(self, action):
        observation, reward, done=self.task.step(np.squeeze(action))
        return observation, reward, done, None

    def reset(self):
        return self.task.reset()

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass
