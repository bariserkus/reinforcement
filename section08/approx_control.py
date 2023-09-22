from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt
from grid_class import standard_grid, print_values, print_policy
from sklearn.kernel_approximation import Nystroem, RBFSampler

GAMMA = 0.9
ALPHA = 0.01
ALL_POSSIBLE_ACTIONS = ("U", "D", "L", "R")
ACTION2INT = {a: i for i, a in enumerate(ALL_POSSIBLE_ACTIONS)}
INT2ONEHOT = np.eye(len(ALL_POSSIBLE_ACTIONS))


def eps_greedy(model, s, eps=0.1):
    p = np.random.random()
    if p < (1 - eps):
        values = model.predict_all_actions(s)
        return ALL_POSSIBLE_ACTIONS[np.argmax(values)]
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)

def one_hot(k):
    return INT2ONEHOT[k]

def merge_state_action(s, a):
    ai = one_hot(ACTION2INT[a])
    return np.concatenate((s, ai))

def gather_samples(g, n_episodes=1000):
    samples = []
    for _ in range(n_episodes):
        s = g.reset()
        samples.append(s)
        while not g.game_over():
            a = np.random.choice(ALL_POSSIBLE_ACTIONS)
            sa = merge_state_action(s, a)
            samples.append(s)

            r = g.move(a)
            s = g.current_state()
    return samples


class Model:
    def __init__(self, g):
        samples = gather_samples(g)
        # self.featurizer = Nystroem()
        self.featurizer = RBFSampler()
        self.featurizer.fit(samples)
        dims = self.featurizer.n_components

        self.w = np.zeros(dims)

    def predict(self, s):
        x = self.featurizer.transform([s])[0]
        return x @ self.w

    def grad(self, s):
        x = self.featurizer.transform([s])[0]
        return x


if __name__ == '__main__':
    grid = standard_grid()

    print("Rewards: ")
    print_values(grid.rewards, grid)

    greedy_policy = {
        (2, 0): "U",
        (1, 0): "U",
        (0, 0): "R",
        (0, 1): "R",
        (0, 2): "R",
        (1, 2): "R",
        (2, 1): "R",
        (2, 2): "R",
        (2, 3): "U",
    }

    model = Model(grid)
    mse_per_episode = []

    n_episodes = 10000
    for it in range(n_episodes):
        if (it + 1) % 100 == 0:
            print(it + 1)

        s = grid.reset()
        Vs = model.predict(s)
        n_steps = 0
        episode_err = 0

        while not grid.game_over():
            a = eps_greedy(greedy_policy, s)
            r = grid.move(a)
            s2 = grid.current_state()

            if grid.is_terminal(s2):
                target = r
            else:
                Vs2 = model.predict(s2)
                target = r + GAMMA * Vs2

            grad = model.grad(s)
            err = target - Vs
            model.w += ALPHA * err * grad

            n_steps += 1
            episode_err += err*err

            s = s2
            Vs = Vs2

        mse = episode_err / n_steps
        mse_per_episode.append(mse)

    plt.plot(mse_per_episode)
    plt.title("MSE per episode")
    plt.show()

    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = model.predict(s)
        else:
            V[s] = 0

    print("Values:")
    print_values(V, grid)

    print("Policy:")
    print_policy(greedy_policy, grid)




