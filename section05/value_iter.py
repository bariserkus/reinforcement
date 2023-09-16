from __future__ import print_function, division
from builtins import range

import numpy as np
from grid_class import standard_grid, ACTION_SPACE, print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9


def get_tp_and_rew(grid):
    tp_ = {}
    rew_ = {}

    for i in range(grid.rows):
        for j in range(grid.cols):
            s_ = (i, j)
            if not grid.is_terminal(s_):
                for a_ in ACTION_SPACE:
                    s2_ = g.get_next_state(s_, a_)
                    tp_[(s_, a_, s2_)] = 1
                    if s2_ in grid.rewards:
                        rew_[(s_, a_, s2_)] = grid.rewards[s2_]

    return tp_, rew_


def eval_pol(grid, initV=None):
    if initV is None:
        V_ = {}
        for s_ in grid.all_states():
            V_[s_] = 0
    else:
        V_ = initV

    # Repeat until convergence
    it = 0
    while True:
        biggest_change = 0
        for s_ in grid.all_states():
            if not grid.is_terminal(s_):
                old_v = V_[s_]
                new_v = float("-inf")
                for a_ in ACTION_SPACE:
                    v = 0
                    for s2_ in grid.all_states():
                        # reward is a function of (s, a, s_prime), 0 if not specified
                        r_ = rew.get((s_, a_, s2_), 0)
                        v += tp.get((s_, a_, s2_), 0) * (r_ + GAMMA * V_[s2_])

                    if v > new_v:
                        new_v = v

                # after getting the new value, update the value table
                V_[s_] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V_[s_]))
        it += 1
        if biggest_change < SMALL_ENOUGH:
            break
    return V_


if __name__ == "__main__":
    g = standard_grid()
    tp, rew = get_tp_and_rew(g)

    policy = {}
    for s in g.actions.keys():
        policy[s] = np.random.choice(ACTION_SPACE)

    print("Initial Policy: ")
    print_policy(policy, g)

    V = {}
    states = g.all_states()
    for s in states:
        V[s] = 0

    V = eval_pol(g, initV=V)

    # find a policy that leads to optimal value function
    for s in g.actions.keys():
        best_a = None
        best_value = float("-inf")
        # loop through all possible actions to find the best current action
        for a in ACTION_SPACE:
            v = 0
            for s2 in g.all_states():
                # reward is a function of (s, a, s'), 0 if not specified
                r = rew.get((s, a, s2), 0)
                v += tp.get((s, a, s2), 0) * (r + GAMMA * V[s2])

            # best_a is the action associated with best_value
            if v > best_value:
                best_value = v
                best_a = a
        policy[s] = best_a

    print("Values:")
    print_values(V, g)

    print("Final Policy")
    print_policy(policy, g)
