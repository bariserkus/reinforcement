from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt
from grid_class import standard_grid, print_policy, print_values

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ("U", "D", "L", "R")


def eps_greedy(pol, s, eps=0.1):
    p = np.random.random()
    if p < (1 - eps):
        return pol[s]
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)


if __name__ == '__main__':
    grid = standard_grid()

    print("Rewards: ")
    print_values(grid.rewards, grid)

    policy = {
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

    V = {}
    states = grid.all_states()
    for s in states:
        V[s] = 0

    deltas = []

    n_episodes = 10000
    for it in range(n_episodes):
        s = grid.reset()

        delta = 0
        while not grid.game_over():
            a = eps_greedy(policy, s)

            r = grid.move(a)
            s_next = grid.current_state()

            v_old = V[s]
            V[s] = V[s] + ALPHA * (r + GAMMA * V[s_next] - V[s])
            delta = max(delta, np.abs(V[s] - v_old))

            s = s_next

            deltas.append(delta)

    plt.plot(deltas)
    plt.show()

    print("Values:")
    print_values(V, grid)
    print("Policy:")
    print_policy(policy, grid)
