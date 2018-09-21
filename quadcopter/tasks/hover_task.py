import numpy as np
from physics_sim import PhysicsSim
from tasks.task import Task

class HoverTask(Task):
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, min_height=1., reward_weights = None, runtime=5.):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            min_height: minimum height to maintain in meters
            runtime: time limit for each episode in seconds
        """
        # Simulation
        self.init_pose = init_pose if init_pose is not None else np.array([0.,0.,10.,0.,0.,0.])
        self.init_velocities = init_velocities if init_velocities is not None else np.zeros(3)
        self.init_angle_velocities = init_angle_velocities if init_angle_velocities is not None else np.zeros(3)
        self.sim = PhysicsSim(self.init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 12 # 6 pose + 6 velocity

        self.action_low = 0    # Min RPM value
        self.action_high = 900 # Max RPM value
        self.action_size = 4   # Num actions (4 rotors)

        # Goal
        self.target_pos = init_pose[0:3] if init_pose is not None else np.array([0., 0., 10.])
        self.reward_weights = reward_weights if reward_weights is not None else np.array([10.,0.,-2.,-1.])

    # Helpful for debugging
    def get_reward_components(self):
        norm = lambda a: np.sum(np.square(a))
        return 1, norm(self.sim.pose[0:3]-self.target_pos), norm(self.sim.v), norm(self.sim.angular_v)

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = np.dot(np.array(self.get_reward_components()), self.reward_weights)
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []

        # Clip rotor speeds to min and max
        rotor_speeds = np.maximum(rotor_speeds, self.action_low*np.ones(self.action_size))
        rotor_speeds = np.minimum(rotor_speeds, self.action_high*np.ones(self.action_size))

        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(np.concatenate((self.sim.pose, self.sim.v, self.sim.angular_v))) # save pose, v and angular_v
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
#         pdb.set_trace()
        state = np.concatenate([np.concatenate((self.sim.pose, self.sim.v, self.sim.angular_v))] * self.action_repeat)
        return state
