# Setup

import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd


def record_quadcopter_episode(agent, filename=None, training=False):
    env = agent.env
    task = env.task
    done = False
    labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
              'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
              'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
    results = {x : [] for x in labels}
    # Run the simulation, and save the results.
    state = agent.reset_episode()
    while True:
        rotor_speeds = agent.act(state, training)
        next_state, reward, done, _ = env.step(rotor_speeds)
        record = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(rotor_speeds)
        for ii in range(len(labels)):
            results[labels[ii]].append(record[ii])
        if done:
            break
    results = pd.DataFrame(data=results)
    if filename is not None:
        results.to_csv(filename)
    return results

def load_quadcopter_episode(filename):
    return pd.DataFrame.from_csv(filename)

def plot_quadcopter_episode(results):
    plt.plot(results['time'], results['x'], label='x')
    plt.plot(results['time'], results['y'], label='y')
    plt.plot(results['time'], results['z'], label='z')
    plt.legend()
    _ = plt.ylim()

    plt.figure()

    plt.plot(results['time'], results['phi'], label='phi')
    plt.plot(results['time'], results['theta'], label='theta')
    plt.plot(results['time'], results['psi'], label='psi')
    plt.legend()
    _ = plt.ylim()

    plt.figure()

    plt.plot(results['time'], results['x_velocity'], label='x_velocity')
    plt.plot(results['time'], results['y_velocity'], label='y_velocity')
    plt.plot(results['time'], results['z_velocity'], label='z_velocity')
    plt.legend()
    _ = plt.ylim()

    angular_v = np.array([results['phi'],results['theta'],results['psi']]).T
    spins = np.sqrt(np.sum(np.square(angular_v),axis=1))
    height_diffs = np.abs(results['z']-results['z'][0])
    vert_v = np.abs(results['z_velocity'])
    # angular_v = np.hstack((np.array(results['phi']).T, np.array(results['theta']).T, np.array(results['psi']).T))
    plt.figure()
    plt.plot(results['time'], height_diffs, label='height_diffs')
    plt.plot(results['time'], vert_v, label='vert_v')
    plt.plot(results['time'], spins, label='spins')
    plt.legend()
    _ = plt.ylim()
