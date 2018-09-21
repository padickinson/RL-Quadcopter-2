import pdb
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from timeit import default_timer as timer
import time

class Agent:
    def __init__(self,
            warmup_episodes = 0,
            episodes_trained = 0,
            train_scores = None,
            test_scores = None,
            best_train_score = -np.inf
            ):
        self.warmup_episodes = warmup_episodes
        self.episodes_trained = episodes_trained
        self.train_scores = train_scores if train_scores is not None else []
        self.test_scores = test_scores if test_scores is not None else []
        self.best_train_score = best_train_score


    def train_step(self):
        raise NotImplementedError()

    def new_episode(self):
        raise NotImplementedError()

    def act(self, state, training):
        raise NotImplementedError()

    def sense(self, s, a, r, s_new, d):
        raise NotImplementedError()

    def train(self,env,n_episodes,n_steps=None,eval_period=None,print_eval=False,render_on_eval=False,render_period=None,reward_window=100):
        start = timer()
        try:
            rewards = deque(maxlen=reward_window)
            for episode in range(1, n_episodes+1):
                if eval_period is not None and episode % eval_period == 0:
                    if print_eval: print('')
                    self.eval(env, n_steps = n_steps, render= render_on_eval, print_eval=print_eval)
                e_start = timer()
                self.new_episode()
                s = env.reset()
                r_sum = 0
                t = 0
                done = False
                while True:
                    if (n_steps is not None and t > n_steps) or done: break

                    if render_period is not None and episode % render_period == 0:
                        env.render()

                    a = self.act(s)
                    # pdb.set_trace()
                    s_new, r, done, _ = env.step(a)
                    self.sense(s, a, r, s_new, done)
                    if episode > self.warmup_episodes:
                        self.train_step()
                    r_sum += r
                    s = s_new
                    t += 1

                rewards.append(float(r_sum))
                self.train_scores.append(float(r_sum))
                if r_sum > self.best_train_score: self.best_train_score = float(r_sum)
                e_end = timer()
                e_time = e_end - e_start
                self.episodes_trained += 1
                print ("\rEp {:>5d}, steps {:>4d}, reward {:>8.5f}, mean {:>8.5f}, best {:>8.5f}, time {:>3.1f}s steps/sec {:>4d}.".format(
                                self.episodes_trained, t, rewards[-1], sum(rewards) / len(rewards), self.best_train_score, e_time, int(t/e_time)),end="")

        except KeyboardInterrupt:
            print ("Training interrupted by user!")

        end = timer()
        t_time = end-start

        print('\nTrained {} episodes. Elapsed time: {:.2f}s. Avg Time per episode: {:.2f}s'.format(episode,t_time,t_time/(episode-1) ))

    def eval(self, env, n_episodes=1, n_steps=1000, render=False, print_eval=False):
        learning_phase = self.learning_phase
        self.learning_phase = False
        rewards = []
        try:
            for episode in range(n_episodes):
                done = False
                s = env.reset()
                reward = 0
                t = 0
                while True:
                    t+=1
                    if render:
                        env.render()
                    a = self.act(s)
                    s_new, r , done, _ = env.step(a)
                    reward += r
                    s = s_new
                    if done or (n_steps is not None and t >= n_steps):
                        break
                rewards.append(reward)
                self.test_scores.append((self.episodes_trained, float(reward)))
                if print_eval: print('Episode {} Steps {} Reward {}'.format(episode+1,t,reward))
        except KeyboardInterrupt:
            print ("Evaluation interrupted by the user!")

        self.learning_phase = learning_phase




    def plot_training_scores(self, smoothing=10):
        num_episodes = self.episodes_trained
        plt.figure()
        # Plot steps
        plt.plot(range(num_episodes),self.train_scores,label='scores')
        plt.plot(range(smoothing,num_episodes),[np.mean(self.train_scores[i-smoothing:i]) for i in range(num_episodes) if i >= smoothing], label='smoothed')
        plt.legend()

    def plot_eval_scores(self, smoothing=10):
        plt.figure()
        # Plot steps
        if len(self.test_scores) == 0:
            print('No scores to plot')
            return
        x = [self.test_scores[0][0]]
        y = [self.test_scores[0][1]]
        n = 1
        for e, s in self.test_scores[1:]:
            if e == x[-1]:
                n+=1
                y[-1] = (n-1)/(n) * y[-1] + 1./n * s
            else:
                x.append(e)
                y.append(s)
                n = 1

        plt.plot(x,y,label='scores')

        # plt.plot(range(smoothing,num_episodes),[np.mean(self.train_scores[i-smoothing:i]) for i in range(num_episodes) if i >= smoothing])
