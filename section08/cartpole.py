from __future__ import print_function, division
from builtins import range

import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt
from grid_class import negative_grid, print_values, print_policy
from sklearn.kernel_approximation import Nystroem, RBFSampler

GAMMA = 0.99
ALPHA = 0.1


def eps_greedy(model, s, eps=0.1):
    p = np.random.random()
    if p < (1 - eps):
        values = model.predict_all_actions(s)
        return np.argmax(values)
    else:
        return model.env.action_space.sample()


def gather_samples(env, n_episodes=5000):
    samples = []
    for _ in range(n_episodes):
        s, info = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            a = env.action_space.sample()
            sa = np.concatenate((s, [a]))
            samples.append(sa)

            s, r, done, truncated, info = env.step(a)
    return samples


class Model:
    def __init__(self, e):
        self.env = e
        samples = gather_samples(e)
        # self.featurizer = Nystroem()
        self.featurizer = RBFSampler()
        self.featurizer.fit(samples)
        dims = self.featurizer.n_components

        self.w = np.zeros(dims)

    def predict(self, s, a):
        sa = sa = np.concatenate((s, [a]))
        x = self.featurizer.transform([sa])[0]
        return x @ self.w

    def predict_all_actions(self, s):
        return [self.predict(s, a) for a in range(self.env.action_space.n)]

    def grad(self, s, a):
        sa = np.concatenate((s, [a]))
        x = self.featurizer.transform([sa])[0]
        return x


def tagent(model, env, n_episodes=20):
    reward_per_episode = np.zeros(n_episodes)
    for it in range(n_episodes):
        done = False
        truncated = False
        episode_reward = 0
        s, info = env.reset()
        while not (done or truncated):
            a = eps_greedy(model, s, eps=0)
            s, r, done, truncated, info = env.step(a)
            episode_reward += r
        reward_per_episode[it] = episode_reward
    return np.mean(reward_per_episode)


def watch_agent(model, env, eps):
    done = False
    truncated = False
    episode_reward = 0
    s, info = env.reset()
    while not (done or truncated):
        a = eps_greedy(model, s, eps=eps)
        s, r, done, truncated, info = env.step(a)
        episode_reward += r
    print("Episode reward:", episode_reward)


if __name__ == '__main__':
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    model = Model(env)
    reward_per_episode = []

    watch_agent(model, env, eps=0)

    n_episodes = 1500
    for it in range(n_episodes):
        s, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        while not (done or truncated):
            a = eps_greedy(model, s)
            s2, r, done, truncated, info = env.step(a)

            if done:
                target = r
            else:
                values = model.predict_all_actions(s2)
                target = r + GAMMA * np.max(values)

            grad = model.grad(s, a)
            err = target - model.predict(s, a)
            model.w += ALPHA * err * grad

            episode_reward += r

            s = s2
        if (it + 1) % 50 == 0:
            print(f"Episode: {it + 1}, Reward: {episode_reward}")

        # early exit
        if it > 20 and np.mean(reward_per_episode[-20:]) == 200:
            print("Early exit")
            break

        reward_per_episode.append(episode_reward)

    # test trained agent
    testreward = tagent(model, env)
    print(f"Average test reward: {testreward}")

    plt.plot(reward_per_episode)
    plt.title("Reward per episode")
    plt.show()

    # watch trained agent
    env = gym.make("CartPole-v1", render_mode="human")
    watch_agent(model, env, eps=0)
