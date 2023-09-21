from __future__ import print_function, division
from builtins import range

import numpy as np
from grid_class import standard_grid, print_policy, print_values

GAMMA = 0.9


def play_game(g, pol, max_steps=20):
    start_states = list(g.actions.keys())
    start_id = np.random.choice(len(start_states))
    g.set_state(start_states[start_id])

    s = g.current_state()

    states = [s]
    rewards = [0]

    steps = 0
    while not g.game_over():
        a = pol[s]
        r = g.move(a)
        next_s = g.current_state()

        states.append(next_s)
        rewards.append(r)

        steps += 1
        if steps >= max_steps:
            break

        s = next_s

    return states, rewards


if __name__ == '__main__':
    grid = standard_grid()

    print("Rewards : ")
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
    returns = {}
    states_all = grid.all_states()

    for s in states_all:
        if s in grid.actions:
            returns[s] = []
        else:
            V[s] = 0

    for _ in range(100):
        states_, rewards = play_game(grid, policy)
        G = 0
        T = len(states_)
        for t in range(T-2, -1, -1):
            s = states_[t]
            r = rewards[t+1]
            G = r + GAMMA * G

            if s not in states_[:t]:
                returns[s].append(G)
                V[s] = np.mean(returns[s])

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)





















