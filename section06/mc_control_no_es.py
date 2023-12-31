from __future__ import print_function, division
from builtins import range

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from grid_class import standard_grid, print_values, print_policy

GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ("U", "D", "L", "R")


def eps_greedy(pol, s, eps=0.1):
    p = np.random.random()
    if p < (1 - eps):
        return pol[s]
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)


def play_game(g, pol, max_steps=20):

    s = grid.reset()

    a = eps_greedy(pol, s)

    states = [s]
    actions = [a]
    rewards = [0]

    for _ in range(max_steps):
        r = g.move(a)
        s = g.current_state()

        rewards.append(r)
        states.append(s)

        if g.game_over():
            break
        else:
            a = pol[s]
            actions.append(a)

  # we want to return:
  # states  = [s(0), s(1), ..., s(T-1), s(T)]
  # actions = [a(0), a(1), ..., a(T-1),     ]
  # rewards = [   0, R(1), ..., R(T-1), R(T)]

    return  states, actions, rewards

def max_dict(d):
    max_val = max(d.values())

    max_keys = [key for key, val in d.items() if val == max_val]

    return np.random.choice(max_keys), max_val

if __name__ == '__main__':

    grid = standard_grid()

    print("Rewards: ")
    print_values(grid.rewards, grid)

    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    Q = {}
    sample_counts = {}
    state_sample_count = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            Q[s] = {}
            sample_counts[s] = {}
            state_sample_count[s] = 0
            for a in ALL_POSSIBLE_ACTIONS:
                Q[s][a] = 0
                sample_counts[s][a] = 0
            else:
                pass

    deltas = []
    for it in range(10000):
        if it % 1000 == 0:
            print(it)

        biggest_change = 0
        states, actions, rewards = play_game(grid, policy)

        states_actions = list(zip(states, actions))

        T = len(states)
        G = 0
        for t in range(T-2, -1, -1):
            s = states[t]
            a = actions[t]

            G = rewards[t+1] + GAMMA * G

            if (s,a) not in states_actions[:t]:
                old_q = Q[s][a]
                sample_counts[s][a] += 1
                lr = 1 / sample_counts[s][a]
                Q[s][a] = old_q + lr * (G - old_q)

                policy[s] = max_dict(Q[s])[0]

                state_sample_count[s] += 1

                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))

        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    print("Final Policy: ")
    print_policy(policy, grid)

    V = {}
    for s, Qs in Q.items():
        V[s] = max_dict(Q[s])[1]

    print("Values: ")
    print_values(V, grid)

    print("state_sample_count:")
    state_sample_count_arr = np.zeros((grid.rows, grid.cols))
    for i in range(grid.rows):
        for j in range(grid.cols):
            if (i, j) in state_sample_count:
                state_sample_count_arr[i, j] = state_sample_count[(i, j)]
    df = pd.DataFrame(state_sample_count_arr)
    print(df)





