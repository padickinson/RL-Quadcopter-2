import matplotlib.pyplot as plt
import numpy as np
import time

class Agent:
    def __init__(self,
            env,
            episodes_trained = 0,
            steps_trained = 0,
            train_scores = None,
            test_scores = None,
            best_train_score = -np.inf,
            best_test_score = -np.inf,
            test_runs_to_average_over = 5,
            test_every_n_episodes = 20
            ):
        self.env=env
        self.episodes_trained = episodes_trained
        self.steps_trained = steps_trained
        self.train_scores = train_scores if train_scores is not None else []
        self.test_scores = test_scores if test_scores is not None else []
        self.best_train_score = best_train_score
        self.best_test_score = best_test_score

        # training parameters
        self.test_runs_to_average_over = test_runs_to_average_over
        self.test_every_n_episodes = test_every_n_episodes

    def step(self, action, reward, next_state, done, training=True):
        raise NotImplementedError()

    def reset_episode(self):
        raise NotImplementedError()

    def act(self, state, training):
        raise NotImplementedError()

    def learn(self):
        raise NotImplementedError()

    def run_episode(self, training=True, debug=False, render=False, log=False):
        start = time.time()
        state = self.reset_episode()  # start a new episode

        score, step = (0.,0)
        while True:
            if render:
                self.env.render()
            action = self.act(state, training)                      # pick an action based on current state
            next_state, reward, done, _ = self.env.step(action)      # step the enviroment
            step += 1
            score += reward
            self.step(action, reward, next_state, done, training)   # step the agent
            if debug:
                print("step: {:4} state: {} action: {} reward: {:0.4f} next_state: {} score: {:0.4f}".format(
                    step, state, action, reward, next_state, score))
            state = next_state                                      # roll over the state
            if done:
                break
        if render:
            self.env.close()

        if training:
            self.episodes_trained += 1
            self.train_scores.append(score)
            if score > self.best_train_score:
                self.best_train_score = score
        else:
            self.test_scores.append((self.episodes_trained, score))
            if score > self.best_test_score:
                self.best_test_score = score

        end = time.time()
        return score, step, end-start

    def train(self, num_episodes=100, output=True):
        for i_episode in range(self.episodes_trained+1, self.episodes_trained+num_episodes+1):
            score, steps, runtime = self.run_episode(training=True)
            if output:
                print("\rTraining Episode: {:4d}  Steps: {:5}  Score: {:>8.2f} (Best: {:>8.2f})  Time: {:>2.2f}s  Steps/Sec: {:>5d}    ".format(
                    i_episode, steps, score, self.best_train_score, runtime, int(steps/runtime) ), end="")
            if i_episode % self.test_every_n_episodes == 0:
                test_run_scores = []
                for j in range(self.test_runs_to_average_over):
                    score, steps, time = self.run_episode(training=False)
                    test_run_scores.append(score)
                if output:
                    print("\n\r== Test Run == Episodes Trained = {:4d}, Average Score over {} runs  = {:7.3f}".format(
                        i_episode, self.test_runs_to_average_over, np.mean(test_run_scores)))

    def plot_training_scores(self, smoothing=10):
        num_episodes = self.episodes_trained
        plt.figure()
        # Plot steps
        plt.plot(range(num_episodes),self.train_scores,label='scores')
        plt.plot(range(smoothing,num_episodes),[np.mean(self.train_scores[i-smoothing:i]) for i in range(num_episodes) if i >= smoothing])


import random

class RandomAgent(Agent):
    def __init__(self, env):
        super().__init__(env)

    def reset_episode(self):
        return self.env.reset()

    def act(self, *args, **kwargs):
        new_thrust = random.gauss(450., 25.)
        return [new_thrust + random.gauss(0., 1.) for x in range(4)]

    def step(self, *args, **kwargs):
        pass

    def learn(self, *args, **kwargs):
        pass
