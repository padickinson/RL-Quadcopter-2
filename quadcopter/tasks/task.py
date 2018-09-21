import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self):
        """Initialize a Task object .
        Params
        ======
        """
        pass

    def step(self, action):
        """Uses action to obtain next state, reward, done."""
        pass

    def reset(self):
        """Reset the sim to start a new episode and return state."""
        pass
